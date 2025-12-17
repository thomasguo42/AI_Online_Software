import cv2
import numpy as np
import torch
from ultralytics import YOLO
import os
import sys
from scipy.spatial.distance import cosine
import torchvision
from tqdm import tqdm
from PIL import Image
from tools.painter import color_list

# Adjust sys.path to include Track-Anything modules
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "tracker")))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "tracker/model")))
from track_anything import TrackingAnything, parse_augment
from tools.painter import mask_painter

# reID model using torchreid
try:
    import torchreid
except ImportError:
    raise ImportError("Please install torchreid: pip install torchreid")

class ReIDModel:
    def __init__(self, model_path):
        # Build OSNet model
        self.model = torchreid.models.build_model(
            name='osnet_x0_25',
            num_classes=1,  # Dummy value, as we only need features
            pretrained=False
        )
        # Load pre-trained weights, ignoring classifier layer
        state_dict = torch.load(model_path, map_location='cpu')
        state_dict = {k: v for k, v in state_dict.items() if not k.startswith('classifier')}
        self.model.load_state_dict(state_dict, strict=False)
        self.model.eval()
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model.to(self.device)
        # Get test transform (second element of tuple)
        _, self.transform = torchreid.data.transforms.build_transforms(
            height=256, width=128, transforms=['random_flip']
        )

    def extract_embedding(self, image):
        # Convert BGR to RGB and apply transforms
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(image)
        image = self.transform(image).unsqueeze(0).to(self.device)
        with torch.no_grad():
            embedding = self.model(image)
        return embedding.cpu().numpy().flatten()

def generate_video_from_frames(frames, output_path, fps=30):
    frames = torch.from_numpy(np.asarray(frames))
    if not os.path.exists(os.path.dirname(output_path)):
        os.makedirs(os.path.dirname(output_path))
    torchvision.io.write_video(output_path, frames, fps=fps, video_codec="libx264")
    return output_path

def main(video_path, output_path, reid_model_path):
    # Set PyTorch memory management to reduce fragmentation
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

    # Parse Track-Anything arguments
    args = parse_augment()
    args.device = "cuda" if torch.cuda.is_available() else "cpu"
    args.sam_model_type = "vit_b"  # Use lighter SAM model
    args.mask_save = False

    # Load models
    yolo_model = YOLO("yolov8n.pt")  # Pre-trained YOLOv8 model
    reid_model = ReIDModel(reid_model_path)  # OSNet model
    sam_checkpoint = "./checkpoints/sam_vit_b_01ec64.pth"
    xmem_checkpoint = "./checkpoints/XMem-s012.pth"
    e2fgvi_checkpoint = "./checkpoints/E2FGVI-HQ-CVPR22.pth"
    tracker = TrackingAnything(sam_checkpoint, xmem_checkpoint, e2fgvi_checkpoint, args)

    # Initialize video capture
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    frames = []
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    cap.release()

    # Initialize tracking state
    trackers = []  # List of (mask, logit, track_id, label)
    embeddings = []  # reID embeddings for each track
    lost_tracks = []  # List of (track_id, embedding, mask, logit, lost_frames, label)
    track_id_counter = 0
    label_counter = 1  # Start labels from 1 for each object
    similarity_threshold = 0.9
    max_lost_frames = 30
    num_colors = len(color_list)  # Number of colors in color_list (updated to match provided list)

    # Process first frame
    first_frame = frames[0]
    results = yolo_model(first_frame, classes=[0])  # Class 0 is 'person'
    for det in results[0].boxes:
        box = det.xyxy[0].cpu().numpy()  # [x_min, y_min, x_max, y_max]
        cropped_img = first_frame[int(box[1]):int(box[3]), int(box[0]):int(box[2])]
        if cropped_img.size == 0:  # Skip empty crops
            continue

        # Extract reID embedding
        embedding = reid_model.extract_embedding(cropped_img)
        embeddings.append(embedding)

        # Initialize SAM for segmentation with integer coordinates
        tracker.samcontroler.sam_controler.reset_image()
        tracker.samcontroler.sam_controler.set_image(first_frame)
        center_x = int(np.round((box[0] + box[2]) / 2))
        center_y = int(np.round((box[1] + box[3]) / 2))
        points = [[center_x, center_y]]  # Center of bbox
        labels = [1]  # Positive prompt
        mask, logit, painted_image = tracker.first_frame_click(
            image=first_frame,
            points=np.array(points),
            labels=np.array(labels),
            multimask=True
        )
        # Convert mask to binary and assign unique label
        binary_mask = (mask > 0).astype(np.uint8) * label_counter

        trackers.append((binary_mask, logit, track_id_counter, label_counter))
        track_id_counter += 1
        label_counter += 1

    # Process subsequent frames
    output_frames = [first_frame.copy()]
    for frame_idx, frame in enumerate(tqdm(frames[1:], desc="Processing frames"), 1):
        painted_frame = frame.copy()
        tracker.xmem.clear_memory()  # Clear XMem memory
        active_trackers = []

        # Combine all tracker masks into a single template mask
        template_mask = np.zeros_like(first_frame[:,:,0], dtype=np.uint8)
        for _, _, tid, label in trackers:
            for t_mask, t_logit, t_tid, t_label in trackers:
                if t_tid == tid:
                    template_mask[t_mask == t_label] = t_label
                    break

        # Track all objects with a single generator call
        if template_mask.max() > 0:  # Only track if there are active objects
            masks, logits, _ = tracker.generator(
                images=[frame],
                template_mask=template_mask
            )
            mask, logit = masks[0], logits[0]
        else:
            mask = np.zeros_like(first_frame[:,:,0], dtype=np.uint8)
            logit = None

        # Update trackers based on mask labels
        for tracker_idx, (prev_mask, prev_logit, tid, label) in enumerate(trackers):
            # Extract mask for this object's label
            object_mask = (mask == label).astype(np.uint8)
            if object_mask.max() == 0:  # Tracker lost the object
                lost_frames = next((lf for t, _, _, _, lf, _ in lost_tracks if t == tid), 0) + 1
                if lost_frames < max_lost_frames:
                    lost_tracks.append((tid, embeddings[tracker_idx], prev_mask, prev_logit, lost_frames, label))
            else:
                painted_frame = mask_painter(painted_frame, object_mask, mask_color=(tid % num_colors))
                active_trackers.append((object_mask, logit, tid, label))

        trackers = active_trackers

        # Detect new objects with YOLOv8
        results = yolo_model(frame, classes=[0])
        unmatched_detections = []
        for det in results[0].boxes:
            box = det.xyxy[0].cpu().numpy()
            cropped_img = frame[int(box[1]):int(box[3]), int(box[0]):int(box[2])]
            if cropped_img.size == 0:  # Skip empty crops
                continue
            new_embedding = reid_model.extract_embedding(cropped_img)

            # Try to match with lost tracks
            match_found = False
            for lost_idx, (tid, lost_embedding, lost_mask, lost_logit, lost_frames, label) in enumerate(lost_tracks[:]):
                similarity = 1 - cosine(new_embedding, lost_embedding)
                if similarity > similarity_threshold:
                    tracker.samcontroler.sam_controler.reset_image()
                    tracker.samcontroler.sam_controler.set_image(frame)
                    center_x = int(np.round((box[0] + box[2]) / 2))
                    center_y = int(np.round((box[1] + box[3]) / 2))
                    points = [[center_x, center_y]]
                    labels = [1]
                    mask, logit, _ = tracker.first_frame_click(
                        image=frame,
                        points=np.array(points),
                        labels=np.array(labels),
                        multimask=True
                    )
                    binary_mask = (mask > 0).astype(np.uint8) * label
                    trackers.append((binary_mask, logit, tid, label))
                    embeddings.append(new_embedding)
                    lost_tracks.pop(lost_idx)
                    painted_frame = mask_painter(painted_frame, binary_mask, mask_color=(tid % num_colors))
                    match_found = True
                    break

            if not match_found:
                unmatched_detections.append((box, new_embedding))

        # Initialize new trackers for unmatched detections
        for box, new_embedding in unmatched_detections:
            tracker.samcontroler.sam_controler.reset_image()
            tracker.samcontroler.sam_controler.set_image(frame)
            center_x = int(np.round((box[0] + box[2]) / 2))
            center_y = int(np.round((box[1] + box[3]) / 2))
            points = [[center_x, center_y]]
            labels = [1]
            mask, logit, _ = tracker.first_frame_click(
                image=frame,
                points=np.array(points),
                labels=np.array(labels),
                multimask=True
            )
            binary_mask = (mask > 0).astype(np.uint8) * label_counter
            trackers.append((binary_mask, logit, track_id_counter, label_counter))
            embeddings.append(new_embedding)
            painted_frame = mask_painter(painted_frame, binary_mask, mask_color=(track_id_counter % num_colors))
            track_id_counter += 1
            label_counter += 1

        output_frames.append(painted_frame)
        torch.cuda.empty_cache()  # Free unused memory

    # Generate output video
    generate_video_from_frames(output_frames, output_path, fps=fps)
    print(f"Output video saved to {output_path}")

if __name__ == "__main__":
    video_path = "4408.mp4"  # Use sample video for testing
    output_path = "./result/track/output_video.mp4"
    reid_model_path = "./checkpoints/osnet_x0_25_msmt17_combineall_256x128_amsgrad_ep150_stp60_lr0.0015_b64_fb10_softmax_labelsmooth_flip_jitter.pth"
    main(video_path, output_path, reid_model_path)