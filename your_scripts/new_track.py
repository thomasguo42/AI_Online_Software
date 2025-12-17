import cv2
import numpy as np
import torch
from ultralytics import YOLO
import os
import sys
from scipy.spatial.distance import cosine
from scipy.optimize import linear_sum_assignment
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
        self.model = torchreid.models.build_model(
            name='osnet_x0_25',
            num_classes=1,
            pretrained=False
        )
        state_dict = torch.load(model_path, map_location='cpu')
        state_dict = {k: v for k, v in state_dict.items() if not k.startswith('classifier')}
        self.model.load_state_dict(state_dict, strict=False)
        self.model.eval()
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model.to(self.device)
        _, self.transform = torchreid.data.transforms.build_transforms(
            height=256, width=128, transforms=['random_flip']
        )

    def extract_embedding(self, image):
        try:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image = Image.fromarray(image)
            image = self.transform(image).unsqueeze(0).to(self.device)
            with torch.no_grad():
                embedding = self.model(image)
            embedding = embedding.cpu().numpy().flatten()
            if np.isnan(embedding).any() or np.isinf(embedding).any():
                return None
            return embedding
        except Exception as e:
            print(f"Error extracting embedding: {e}")
            return None

def generate_video_from_frames(frames, output_path, fps=30):
    frames = torch.from_numpy(np.asarray(frames))
    if not os.path.exists(os.path.dirname(output_path)):
        os.makedirs(os.path.dirname(output_path))
    torchvision.io.write_video(output_path, frames, fps=fps, video_codec="libx264")
    return output_path

def compute_cost_matrix(detections, tracks):
    cost_matrix = np.zeros((len(detections), len(tracks)))
    for i, det in enumerate(detections):
        for j, track in enumerate(tracks):
            if det['embedding'] is None or track['embedding'] is None:
                cost_matrix[i, j] = 1  # High cost
                continue
            emb_sim = 1 - cosine(det['embedding'], track['embedding'])
            if np.isnan(emb_sim):
                cost_matrix[i, j] = 1  # High cost
            else:
                cost_matrix[i, j] = 1 - emb_sim
    return cost_matrix

def mask_to_box(mask):
    y, x = np.where(mask > 0)
    if len(x) == 0 or len(y) == 0:
        return None
    x1, y1, x2, y2 = x.min(), y.min(), x.max(), y.max()
    return np.array([x1, y1, x2, y2])

def dilate_mask(mask, kernel_size=5):
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    dilated_mask = cv2.dilate(mask, kernel, iterations=1)
    return dilated_mask

def main(video_path, output_path, reid_model_path):
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    args = parse_augment()
    args.device = "cuda" if torch.cuda.is_available() else "cpu"
    args.sam_model_type = "vit_b"
    args.mask_save = False
    similarity_threshold = 0.8

    # Load models
    yolo_model = YOLO("yolov8n.pt")
    reid_model = ReIDModel(reid_model_path)
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

    # Process first frame
    first_frame = frames[0]
    results = yolo_model(first_frame, classes=[0])  # Class 0 for persons
    detections = []
    for det in results[0].boxes:
        box = det.xyxy[0].cpu().numpy()
        cropped_img = first_frame[int(box[1]):int(box[3]), int(box[0]):int(box[2])]
        if cropped_img.size == 0:
            continue
        embedding = reid_model.extract_embedding(cropped_img)
        if embedding is not None:
            detections.append({'box': box, 'embedding': embedding})

    # Display detections and get user input
    print("Detected persons in the first frame:")
    for i, det in enumerate(detections):
        print(f"Index {i}: Box {det['box']}")
    input_str = input("Enter the indices of the two people to track (e.g., '0 1'): ")
    try:
        indices = list(map(int, input_str.split()))
        if len(indices) != 2 or any(i < 0 or i >= len(detections) for i in indices):
            raise ValueError
    except ValueError:
        print("Invalid input. Please enter exactly two valid indices.")
        sys.exit(1)

    # Initialize tracks for selected detections
    selected_detections = [detections[i] for i in indices]
    tracks = []
    combined_mask = np.zeros_like(first_frame[:,:,0], dtype=np.uint8)
    for i, det in enumerate(selected_detections):
        tracker.samcontroler.sam_controler.reset_image()
        tracker.samcontroler.sam_controler.set_image(first_frame)
        box = det['box']
        # Select multiple points: top, center, bottom
        points = [
            [int((box[0] + box[2]) / 2), int(box[1] + 0.25 * (box[3] - box[1]))],  # Top
            [int((box[0] + box[2]) / 2), int((box[1] + box[3]) / 2)],             # Center
            [int((box[0] + box[2]) / 2), int(box[3] - 0.25 * (box[3] - box[1]))]  # Bottom
        ]
        labels = [1, 1, 1]  # All points are positive
        mask, logit, _ = tracker.first_frame_click(
            image=first_frame,
            points=np.array(points),
            labels=np.array(labels),
            multimask=True
        )
        binary_mask = (mask > 0).astype(np.uint8) * (i + 1)  # Labels 1 and 2
        binary_mask = dilate_mask(binary_mask)  # Dilate to cover more area
        combined_mask[binary_mask > 0] = i + 1
        tracks.append({
            'id': i,
            'embedding': det['embedding'],
            'box': box,
            'mask': binary_mask,
            'label': i + 1
        })

    # Initialize tracker with combined mask
    tracker.xmem.track(first_frame, combined_mask)

    # Paint first frame
    painted_frame = first_frame.copy()
    for track in tracks:
        binary_mask = track['mask']
        color_index = track['id'] % len(color_list)
        painted_frame = mask_painter(painted_frame, binary_mask, mask_color=color_index)
        box = track['box'].astype(int)
        color = color_list[color_index]
        cv2.rectangle(painted_frame, (box[0], box[1]), (box[2], box[3]), color, 2)
        cv2.putText(painted_frame, f"ID: {track['id']}", (box[0], box[1] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    output_frames = [painted_frame]

    # Process subsequent frames
    for frame_idx, frame in enumerate(tqdm(frames[1:], desc="Processing frames"), 1):
        mask, _, _ = tracker.xmem.track(frame)  # Get updated combined mask
        painted_frame = frame.copy()

        # Update tracks with tracked masks and boxes
        for track in tracks:
            label = track['label']
            binary_mask = (mask == label).astype(np.uint8)
            binary_mask = dilate_mask(binary_mask)  # Dilate to cover more area
            box = mask_to_box(binary_mask)
            if box is not None:
                track['mask'] = binary_mask
                track['box'] = box
            else:
                # Fallback to last known box if mask is lost
                continue

            # Paint mask and draw box
            color_index = track['id'] % len(color_list)
            painted_frame = mask_painter(painted_frame, binary_mask, mask_color=color_index)
            box = track['box'].astype(int)
            color = color_list[color_index]
            cv2.rectangle(painted_frame, (box[0], box[1]), (box[2], box[3]), color, 2)
            cv2.putText(painted_frame, f"ID: {track['id']}", (box[0], box[1] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        output_frames.append(painted_frame)
        torch.cuda.empty_cache()

    # Generate output video
    generate_video_from_frames(output_frames, output_path, fps=fps)
    print(f"Output video saved to {output_path}")

if __name__ == "__main__":
    video_path = "4409.mp4"  # Replace with your video path
    output_path = "./result/track/output_video.mp4"
    reid_model_path = "./checkpoints/osnet_x0_25_msmt17_combineall_256x128_amsgrad_ep150_stp60_lr0.0015_b64_fb10_softmax_labelsmooth_flip_jitter.pth"  # Replace with your reID model path
    main(video_path, output_path, reid_model_path)