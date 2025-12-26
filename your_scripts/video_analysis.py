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
from .tools.painter import color_list
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.linear_model import LinearRegression
import torchreid
import logging
import gc
import subprocess
import tempfile

# Adjust sys.path for Track-Anything modules
base_dir = os.getcwd()
sys.path.append(os.path.abspath(os.path.join(base_dir, "tracker")))
sys.path.append(os.path.abspath(os.path.join(base_dir, "tracker/model")))
from .track_anything import TrackingAnything, parse_augment
from .tools.painter import mask_painter
from .camera_corrections import ImprovedCameraStabilizer, transform_points

# ReID model class (unchanged)
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

# Utility functions (unchanged)
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
                cost_matrix[i, j] = 1
                continue
            emb_sim = 1 - cosine(det['embedding'], track['embedding'])
            if np.isnan(emb_sim):
                cost_matrix[i, j] = 1
            else:
                cost_matrix[i, j] = 1 - emb_sim
    return cost_matrix

def mask_to_box(mask, margin=20):
    y, x = np.where(mask > 0)
    if len(x) == 0 or len(y) == 0:
        return None
    x1, y1, x2, y2 = x.min(), y.min(), x.max(), y.max()
    x1 = max(0, x1 - margin)
    y1 = max(0, y1 - margin)
    x2 = min(mask.shape[1], x2 + margin)
    y2 = min(mask.shape[0], y2 + margin)
    return np.array([x1, y1, x2, y2])

def dilate_mask(mask, kernel_size=9, iterations=2):
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    dilated_mask = cv2.dilate(mask, kernel, iterations=iterations)
    return dilated_mask

def fill_with_linear_regression(series, c, jump_threshold=20):
    series = pd.Series(series).copy()
    diff = series.diff().abs()
    threshold = jump_threshold / c
    erratic = diff > threshold
    series[erratic] = np.nan
    valid_indices = series.index[~series.isna()]
    if len(valid_indices) < 2:
        median = series.median() if not series.isna().all() else 0
        return series.fillna(median).rolling(window=5, center=True, min_periods=1).mean()
    gaps = []
    start = None
    for i in range(len(series)):
        if pd.isna(series[i]) and start is None:
            start = i
        elif (not pd.isna(series[i]) or i == len(series) - 1) and start is not None:
            end = i if not pd.isna(series[i]) else i + 1
            gaps.append((start, end))
            start = None
    for start, end in gaps:
        before = valid_indices[valid_indices < start]
        after = valid_indices[valid_indices >= end]
        if len(before) == 0 or len(after) == 0:
            continue
        before_idx = before[-1]
        after_idx = after[0]
        X = np.array([[before_idx], [after_idx]])
        y = np.array([series[before_idx], series[after_idx]])
        model = LinearRegression()
        model.fit(X, y)
        gap_indices = np.array([[i] for i in range(start, min(end, len(series)))])
        predicted = model.predict(gap_indices)
        series[start:end] = predicted
    series = series.fillna(method='ffill').fillna(method='bfill')
    series = series.rolling(window=5, center=True, min_periods=1).mean()
    return series


def _apply_transform_to_track(track, transform):
    """Apply camera stabilization transform to keypoints and bounding boxes."""
    if 'keypoints' in track and track['keypoints'] is not None:
        keypoints = np.asarray(track['keypoints'], dtype=np.float32)
        if keypoints.ndim == 2 and keypoints.shape[1] >= 2:
            xy = keypoints[:, :2]
            transformed_xy = transform_points(xy, transform)
            if keypoints.shape[1] > 2:
                keypoints[:, :2] = transformed_xy
            else:
                keypoints = transformed_xy
            track['keypoints'] = keypoints

    if 'box' in track and track['box'] is not None:
        box = np.asarray(track['box'], dtype=np.float32)
        if box.size == 4:
            corners = np.array([
                [box[0], box[1]],
                [box[2], box[3]]
            ], dtype=np.float32)
            transformed_corners = transform_points(corners, transform)
            track['box'] = np.array([
                float(transformed_corners[0, 0]),
                float(transformed_corners[0, 1]),
                float(transformed_corners[1, 0]),
                float(transformed_corners[1, 1])
            ])

    if 'box_history' in track and track['box_history']:
        transformed_history = []
        for hist_box in track['box_history']:
            hist_box = np.asarray(hist_box, dtype=np.float32)
            if hist_box.size != 4:
                transformed_history.append(hist_box)
                continue
            corners = np.array([
                [hist_box[0], hist_box[1]],
                [hist_box[2], hist_box[3]]
            ], dtype=np.float32)
            transformed_corners = transform_points(corners, transform)
            transformed_history.append(np.array([
                float(transformed_corners[0, 0]),
                float(transformed_corners[0, 1]),
                float(transformed_corners[1, 0]),
                float(transformed_corners[1, 1])
            ]))
        track['box_history'] = transformed_history

def process_video_and_extract_data(tracks_per_frame, source):
    left_xdata = {k: [] for k in range(17)}
    left_ydata = {k: [] for k in range(17)}
    right_xdata = {k: [] for k in range(17)}
    right_ydata = {k: [] for k in range(17)}
    checker_list = []
    video_angle = ''
    
    # Find the first frame with 'keypoints' for both tracks
    for tracks in tracks_per_frame:
        if len(tracks) == 2 and all('keypoints' in t for t in tracks):
            track0, track1 = tracks
            values = [
                track0['keypoints'][15][0], track0['keypoints'][16][0],
                track1['keypoints'][15][0], track1['keypoints'][16][0]
            ]
            sorted_values = sorted(values, reverse=True)
            b = sorted_values[1]
            a = sorted_values[2]
            c = abs((b - a) / 4)
            # Calculate video_angle based on box areas
            left_box_area = (track0['box'][2] - track0['box'][0]) * (track0['box'][3] - track0['box'][1])
            right_box_area = (track1['box'][2] - track1['box'][0]) * (track1['box'][3] - track1['box'][1])
            if left_box_area >= 1.75 * right_box_area:
                video_angle = 'left'
            elif right_box_area >= 1.75 * left_box_area:
                video_angle = 'right'
            else:
                video_angle = 'middle'
            break
    else:
        raise ValueError("No frame with 'keypoints' for both tracks found in the video")
    
    # Extract data for all frames
    for tracks in tracks_per_frame:
        try:
            if len(tracks) != 2 or not all('keypoints' in t for t in tracks):
                for j in range(17):
                    left_xdata[j].append(np.nan)
                    left_ydata[j].append(np.nan)
                    right_xdata[j].append(np.nan)
                    right_ydata[j].append(np.nan)
                continue
            center_x0 = (tracks[0]['box'][0] + tracks[0]['box'][2]) / 2
            center_x1 = (tracks[1]['box'][0] + tracks[1]['box'][2]) / 2
            left_track = tracks[0] if center_x0 < center_x1 else tracks[1]
            right_track = tracks[1] if center_x0 < center_x1 else tracks[0]
            for track, xdata, ydata, is_left in [(left_track, left_xdata, left_ydata, True),
                                                (right_track, right_xdata, right_ydata, False)]:
                keypoints = track['keypoints']
                swapped = keypoints.copy()
                for pair in [(1, 2), (3, 4), (5, 6), (7, 8), (9, 10), (11, 12), (13, 14), (15, 16)]:
                    kp1, kp2 = pair
                    if is_left and keypoints[kp1][0] > keypoints[kp2][0]:
                        swapped[kp1], swapped[kp2] = keypoints[kp2], keypoints[kp1]
                    elif not is_left and keypoints[kp1][0] < keypoints[kp2][0]:
                        swapped[kp1], swapped[kp2] = keypoints[kp2], keypoints[kp1]
                for j in range(17):
                    xdata[j].append(swapped[j][0] / c)
                    ydata[j].append(swapped[j][1] / c)
        except Exception as e:
            for j in range(17):
                left_xdata[j].append(np.nan)
                left_ydata[j].append(np.nan)
                right_xdata[j].append(np.nan)
                right_ydata[j].append(np.nan)
            continue
    
    # Apply linear regression to fill gaps
    for k in range(17):
        left_xdata[k] = fill_with_linear_regression(left_xdata[k], c).tolist()
        left_ydata[k] = fill_with_linear_regression(left_ydata[k], c).tolist()
        right_xdata[k] = fill_with_linear_regression(right_xdata[k], c).tolist()
        right_ydata[k] = fill_with_linear_regression(right_ydata[k], c).tolist()
    
    return left_xdata, left_ydata, right_xdata, right_ydata, c, checker_list, video_angle

def save_data_to_csv(left_xdata, left_ydata, right_xdata, right_ydata, c, checker_list, video_angle, output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    left_x_df = pd.DataFrame(left_xdata)
    left_y_df = pd.DataFrame(left_ydata)
    right_x_df = pd.DataFrame(right_xdata)
    right_y_df = pd.DataFrame(right_ydata)
    meta_df = pd.DataFrame({'c': [c], 'checker_list': [str(checker_list)], 'video_angle': [video_angle]})
    left_x_df.to_csv(os.path.join(output_dir, 'left_xdata.csv'), index=False)
    left_y_df.to_csv(os.path.join(output_dir, 'left_ydata.csv'), index=False)
    right_x_df.to_csv(os.path.join(output_dir, 'right_xdata.csv'), index=False)
    right_y_df.to_csv(os.path.join(output_dir, 'right_ydata.csv'), index=False)
    meta_df.to_csv(os.path.join(output_dir, 'meta.csv'), index=False)

def load_data_from_csv(input_dir):
    left_x_df = pd.read_csv(os.path.join(input_dir, 'left_xdata.csv'))
    left_y_df = pd.read_csv(os.path.join(input_dir, 'left_ydata.csv'))
    right_x_df = pd.read_csv(os.path.join(input_dir, 'right_xdata.csv'))
    right_y_df = pd.read_csv(os.path.join(input_dir, 'right_ydata.csv'))
    meta_df = pd.read_csv(os.path.join(input_dir, 'meta.csv'))
    left_xdata = {int(k): left_x_df[k].tolist() for k in left_x_df.columns}
    left_ydata = {int(k): left_y_df[k].tolist() for k in left_y_df.columns}
    right_xdata = {int(k): right_x_df[k].tolist() for k in right_x_df.columns}
    right_ydata = {int(k): right_y_df[k].tolist() for k in right_y_df.columns}
    c = meta_df['c'][0]
    checker_list = eval(meta_df['checker_list'][0]) if meta_df['checker_list'][0] else []
    video_angle = meta_df['video_angle'][0]
    return left_xdata, left_ydata, right_xdata, right_ydata, c, checker_list, video_angle

def process_first_frame(video_path, output_dir):
    yolo_model = YOLO("yolov8l.pt")
    cap = cv2.VideoCapture(video_path)
    ret, frame = cap.read()
    if not ret:
        cap.release()
        raise ValueError("Could not read video frame")
    cap.release()
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = yolo_model(frame_rgb, classes=[0])
    detections = []
    reid_model = ReIDModel(os.path.join(os.getcwd(), "your_scripts", "checkpoints", "osnet_x0_25_msmt17_combineall_256x128_amsgrad_ep150_stp60_lr0.0015_b64_fb10_softmax_labelsmooth_flip_jitter.pth"))
    for det in results[0].boxes:
        box = det.xyxy[0].cpu().numpy()
        cropped_img = frame_rgb[int(box[1]):int(box[3]), int(box[0]):int(box[2])]
        if cropped_img.size == 0:
            continue
        embedding = reid_model.extract_embedding(cropped_img)
        if embedding is not None:
            detections.append({'box': box, 'embedding': embedding})
    display_frame = frame_rgb.copy()
    for i, det in enumerate(detections):
        box = det['box'].astype(int)
        cv2.rectangle(display_frame, (box[0], box[1]), (box[2], box[3]), (0, 255, 0), 2)
        cv2.putText(display_frame, f"ID: {i}", (box[0], box[1] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    detection_image_path = os.path.join(output_dir, 'detection.png')
    plt.figure(figsize=(10, 6))
    plt.imshow(display_frame)
    plt.axis('off')
    plt.savefig(detection_image_path, bbox_inches='tight')
    plt.close()
    return detection_image_path, detections

def compress_video(input_path, output_dir, resolution="1280:720", bitrate="2000k"):
    """Compress the input video to a lower resolution and bitrate."""
    # Ensure output filename has .mp4 extension
    base_name = os.path.basename(input_path)
    if not base_name.lower().endswith('.mp4'):
        temp_file = os.path.join(output_dir, f"compressed_{base_name}.mp4")
    else:
        temp_file = os.path.join(output_dir, f"compressed_{base_name}")
    try:
        cmd = [
            "ffmpeg",
            "-y",  # Overwrite output file if it exists
            "-i", input_path,
            "-vf", f"scale={resolution}",
            "-c:v", "libx264",
            "-b:v", bitrate,
            "-c:a", "aac",
            "-b:a", "128k",
            temp_file
        ]
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        return temp_file
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"FFmpeg compression failed: {e.stderr}")
    except Exception as e:
        raise RuntimeError(f"Error during compression: {str(e)}")

def main(
    video_path,
    output_path,
    reid_model_path,
    csv_output_dir,
    selected_indexes,
    fps=30,
    yolo_model_path="yolov8l.pt",
    pose_model_path="yolov8l-pose.pt",
    sam_checkpoint=None,
    sam_model_type="vit_b",
    xmem_checkpoint=None,
    e2fgvi_checkpoint=None,
    yolo_conf=0.7,
):
    torch.cuda.empty_cache()
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

    # Check video resolution and compress if large
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Could not open video file {video_path}")
    original_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    original_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    original_fps = cap.get(cv2.CAP_PROP_FPS)
    cap.release()
    LARGE_VIDEO_THRESHOLD = (1920, 1080)
    is_large_video = original_width >= LARGE_VIDEO_THRESHOLD[0] or original_height >= LARGE_VIDEO_THRESHOLD[1]
    processing_path = video_path
    compressed_path = None
    scale_factor = 1.0
    if is_large_video:
        print(f"Large video detected ({original_width}x{original_height}). Compressing to 1280x720.")
        compressed_path = compress_video(video_path, os.path.dirname(video_path), resolution="1280:720", bitrate="2000k")
        processing_path = compressed_path
        scale_factor = 720 / original_height
    else:
        print(f"Video resolution ({original_width}x{original_height}) is not large. Using original video.")
    fps = fps if fps is not None else original_fps

    # Initialize tracking
    original_argv = sys.argv
    sys.argv = ['']
    args = parse_augment()
    sys.argv = original_argv
    args.device = "cuda" if torch.cuda.is_available() else "cpu"
    args.sam_model_type = sam_model_type
    args.mask_save = False
    similarity_threshold = 0.8
    yolo_model = YOLO(yolo_model_path)
    pose_model = YOLO(pose_model_path)
    reid_model = ReIDModel(reid_model_path)
    if sam_checkpoint is None:
        sam_checkpoint = os.path.join(os.getcwd(), "your_scripts", "checkpoints", "sam_vit_b_01ec64.pth")
    if xmem_checkpoint is None:
        xmem_checkpoint = os.path.join(os.getcwd(), "your_scripts", "checkpoints", "XMem-s012.pth")
    if e2fgvi_checkpoint is None:
        e2fgvi_checkpoint = os.path.join(os.getcwd(), "your_scripts", "checkpoints", "E2FGVI-HQ-CVPR22.pth")
    tracker = TrackingAnything(sam_checkpoint, xmem_checkpoint, e2fgvi_checkpoint, args)
    stabilizer = ImprovedCameraStabilizer(use_strip_detection=True)

    # Process frames incrementally
    cap = cv2.VideoCapture(processing_path)
    if not cap.isOpened():
        raise ValueError(f"Could not open video file {processing_path}")
    ret, first_frame = cap.read()
    if not ret:
        raise ValueError(f"No frames read from processing video {processing_path}")
    first_frame_rgb = cv2.cvtColor(first_frame, cv2.COLOR_BGR2RGB)
    tracks_per_frame = []

    # First frame processing
    results = yolo_model(first_frame_rgb, classes=[0], conf=yolo_conf)
    detections = []
    for det in results[0].boxes:
        box = det.xyxy[0].cpu().numpy()
        if int(box[3]) > int(box[1]) and int(box[2]) > int(box[0]):
            cropped_img = first_frame_rgb[int(box[1]):int(box[3]), int(box[0]):int(box[2])]
            if cropped_img.size > 0:
                embedding = reid_model.extract_embedding(cropped_img)
                if embedding is not None:
                    detections.append({'box': box, 'embedding': embedding})
    if not all(0 <= i < len(detections) for i in selected_indexes):
        raise ValueError("Invalid selected indexes")
    pose_results = pose_model(first_frame_rgb)
    poses = []
    for i, box in enumerate(pose_results[0].boxes):
        box = box.xyxy[0].cpu().numpy()
        keypoints = pose_results[0].keypoints[i].data.cpu().numpy()
        if keypoints.size > 0:
            if keypoints.ndim == 3 and keypoints.shape[1] == 17 and keypoints.shape[2] == 3:
                poses.append({'box': box, 'keypoints': keypoints[0]})
            else:
                print(f"Malformed keypoints for box {i} in first frame")
                if keypoints.ndim == 2 and keypoints.shape[0] == 17:
                    keypoints = np.pad(keypoints, ((0, 0), (0, 1)), mode='constant')
                    poses.append({'box': box, 'keypoints': keypoints})
                else:
                    continue
    selected_detections = [detections[i] for i in selected_indexes]
    tracks = []
    combined_mask = np.zeros_like(first_frame_rgb[:,:,0], dtype=np.uint8)
    for i, det in enumerate(selected_detections):
        tracker.samcontroler.sam_controler.reset_image()
        tracker.samcontroler.sam_controler.set_image(first_frame_rgb)
        box = det['box']
        matched_pose = None
        for pose in poses:
            pose_box = pose['box']
            x1 = max(box[0], pose_box[0])
            y1 = max(box[1], pose_box[1])
            x2 = min(box[2], pose_box[2])
            y2 = min(box[3], pose_box[3])
            if x1 < x2 and y1 < y2:
                intersection_area = (x2 - x1) * (y2 - y1)
                box_area = (box[2] - box[0]) * (box[3] - box[1])
                if intersection_area / box_area > 0.5:
                    matched_pose = pose
                    break
        if matched_pose:
            keypoints = matched_pose['keypoints']
            torso_keypoints = [kp for idx, kp in enumerate(keypoints) if idx in [5, 6, 11, 12] and len(kp) >= 3 and kp[2] > 0.5]
            if not torso_keypoints:
                torso_keypoints = [kp for idx, kp in enumerate(keypoints) if idx in [5, 6, 11, 12] and len(kp) >= 2]
            if torso_keypoints:
                torso_x = int(np.mean([kp[0] for kp in torso_keypoints]))
                torso_y = int(np.mean([kp[1] for kp in torso_keypoints]))
                width = box[2] - box[0]
                height = box[3] - box[1]
                points = [
                    [torso_x, torso_y],
                    [int(torso_x - width * 0.1), torso_y],
                    [int(torso_x + width * 0.1), torso_y],
                    [torso_x, int(torso_y - height * 0.1)],
                    [torso_x, int(torso_y + height * 0.1)]
                ]
            else:
                print(f"Warning: No torso keypoints found for fencer {i}, falling back to bounding box points")
                width = box[2] - box[0]
                height = box[3] - box[1]
                mid_x = int((box[0] + box[2]) / 2)
                mid_y = int((box[1] + box[3]) / 2)
                points = [
                    [mid_x, mid_y],
                    [int(mid_x - width * 0.2), mid_y],
                    [int(mid_x + width * 0.2), mid_y],
                    [mid_x, int(box[1] + height * 0.3)],
                    [mid_x, int(box[1] + height * 0.7)]
                ]
        else:
            print(f"Warning: No matching pose found for fencer {i}, using bounding box points")
            width = box[2] - box[0]
            height = box[3] - box[1]
            mid_x = int((box[0] + box[2]) / 2)
            mid_y = int((box[1] + box[3]) / 2)
            points = [
                [mid_x, mid_y],
                [int(mid_x - width * 0.2), mid_y],
                [int(mid_x + width * 0.2), mid_y],
                [mid_x, int(box[1] + height * 0.3)],
                [mid_x, int(box[1] + height * 0.7)]
            ]
        negative_points = [
            [int(box[0] - width * 0.2), int(box[1] - height * 0.2)],
            [int(box[2] + width * 0.2), int(box[1] - height * 0.2)],
            [int(box[0] - width * 0.2), int(box[3] + height * 0.2)],
            [int(box[2] + width * 0.2), int(box[3] + height * 0.2)]
        ]
        negative_points = [
            [max(0, min(first_frame_rgb.shape[1] - 1, x)), max(0, min(first_frame_rgb.shape[0] - 1, y))]
            for x, y in negative_points
        ]
        points.extend(negative_points)
        labels = [1, 1, 1, 1, 1] + [0, 0, 0, 0]
        print(f"Click points for Fencer {i}:")
        for idx, (x, y) in enumerate(points):
            label_type = "Positive" if labels[idx] == 1 else "Negative"
            print(f"  Point {idx}: ({x}, {y}) - {label_type}")
        mask, logit, _ = tracker.first_frame_click(
            image=first_frame_rgb,
            points=np.array(points),
            labels=np.array(labels),
            multimask=True
        )
        binary_mask = (mask > 0).astype(np.uint8) * (i + 1)
        binary_mask = dilate_mask(binary_mask, kernel_size=5, iterations=1)
        combined_mask[binary_mask > 0] = i + 1
        tracks.append({
            'id': i,
            'embedding': det['embedding'],
            'box': box,
            'mask': binary_mask,
            'label': i + 1,
            'box_history': [box]
        })
    tracker.xmem.track(first_frame_rgb, combined_mask)
    pose_results = pose_model(first_frame_rgb)
    poses = []
    for i, box in enumerate(pose_results[0].boxes):
        box = box.xyxy[0].cpu().numpy()
        keypoints = pose_results[0].keypoints[i].data.cpu().numpy()
        if keypoints.size > 0:
            if keypoints.ndim == 3 and keypoints.shape[1] == 17 and keypoints.shape[2] == 3:
                poses.append({'box': box, 'keypoints': keypoints[0]})
            else:
                print(f"Malformed keypoints for box {i} in first frame")
                if keypoints.ndim == 2 and keypoints.shape[0] == 17:
                    keypoints = np.pad(keypoints, ((0, 0), (0, 1)), mode='constant')
                    poses.append({'box': box, 'keypoints': keypoints})
                else:
                    continue
    assigned_poses = {}
    for pose in poses:
        counts = {1: 0, 2: 0}
        for kp in pose['keypoints']:
            if len(kp) >= 3 and kp[2] > 0.5:
                x, y = int(kp[0]), int(kp[1])
                if 0 <= x < first_frame_rgb.shape[1] and 0 <= y < first_frame_rgb.shape[0]:
                    label = combined_mask[y, x]
                    if label in counts:
                        counts[label] += 1
            elif len(kp) == 2:
                x, y = int(kp[0]), int(kp[1])
                if 0 <= x < first_frame_rgb.shape[1] and 0 <= y < first_frame_rgb.shape[0]:
                    label = combined_mask[y, x]
                    if label in counts:
                        counts[label] += 1
        if counts:
            assigned_label = max(counts, key=counts.get)
            if counts[assigned_label] >= 3:
                if assigned_label not in assigned_poses or counts[assigned_label] > assigned_poses[assigned_label]['count']:
                    assigned_poses[assigned_label] = {'pose': pose, 'count': counts[assigned_label]}
    for track in tracks:
        label = track['label']
        if label in assigned_poses:
            track['keypoints'] = assigned_poses[label]['pose']['keypoints']
    first_transform = stabilizer.process_frame(first_frame, [det['box'] for det in selected_detections])
    for track in tracks:
        _apply_transform_to_track(track, first_transform)
    tracks_per_frame.append([track.copy() for track in tracks])

    # Process remaining frames
    for frame_idx in tqdm(range(1, int(cap.get(cv2.CAP_PROP_FRAME_COUNT))), desc="Processing frames"):
        ret, frame = cap.read()
        if not ret:
            break
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mask, _, _ = tracker.xmem.track(frame_rgb)
        results = yolo_model(frame_rgb, classes=[0])
        detections = []
        for det in results[0].boxes:
            box = det.xyxy[0].cpu().numpy()
            cropped_img = frame_rgb[int(box[1]):int(box[3]), int(box[0]):int(box[2])]
            if cropped_img.size == 0:
                continue
            embedding = reid_model.extract_embedding(cropped_img)
            if embedding is not None:
                detections.append({'box': box, 'embedding': embedding})
        if detections and tracks:
            cost_matrix = compute_cost_matrix(detections, tracks)
            row_ind, col_ind = linear_sum_assignment(cost_matrix)
            matched_detections = {tracks[j]['label']: detections[i] for i, j in zip(row_ind, col_ind) if cost_matrix[i, j] < 0.2}
        pose_results = pose_model(frame_rgb)
        poses = []
        for i, box in enumerate(pose_results[0].boxes):
            box = box.xyxy[0].cpu().numpy()
            keypoints = pose_results[0].keypoints[i].data.cpu().numpy()
            if keypoints.size > 0:
                if keypoints.ndim == 3 and keypoints.shape[1] == 17 and keypoints.shape[2] == 3:
                    poses.append({'box': box, 'keypoints': keypoints[0]})
                else:
                    print(f"Malformed keypoints for box {i} in frame {frame_idx}")
                    if keypoints.ndim == 2 and keypoints.shape[0] == 17:
                        keypoints = np.pad(keypoints, ((0, 0), (0, 1)), mode='constant')
                        poses.append({'box': box, 'keypoints': keypoints})
                    else:
                        continue
        assigned_poses = {}
        for pose in poses:
            counts = {1: 0, 2: 0}
            for kp in pose['keypoints']:
                if len(kp) >= 3 and kp[2] > 0.5:
                    x, y = int(kp[0]), int(kp[1])
                    if 0 <= x < frame_rgb.shape[1] and 0 <= y < frame_rgb.shape[0]:
                        label = mask[y, x]
                        if label in counts:
                            counts[label] += 1
                elif len(kp) == 2:
                    x, y = int(kp[0]), int(kp[1])
                    if 0 <= x < frame_rgb.shape[1] and 0 <= y < frame_rgb.shape[0]:
                        label = mask[y, x]
                        if label in counts:
                            counts[label] += 1
            if counts:
                assigned_label = max(counts, key=counts.get)
                if counts[assigned_label] >= 3:
                    if assigned_label not in assigned_poses or counts[assigned_label] > assigned_poses[assigned_label]['count']:
                        assigned_poses[assigned_label] = {'pose': pose, 'count': counts[assigned_label]}
        current_tracks = []
        for track in tracks:
            label = track['label']
            binary_mask = (mask == label).astype(np.uint8)
            binary_mask = dilate_mask(binary_mask)
            box = mask_to_box(binary_mask)
            if box is not None:
                track_copy = track.copy()
                track_copy['mask'] = binary_mask
                if label in matched_detections:
                    yolo_box = matched_detections[label]['box']
                    x1 = min(box[0], yolo_box[0])
                    y1 = min(box[1], yolo_box[1])
                    x2 = max(box[2], yolo_box[2])
                    y2 = max(box[3], yolo_box[3])
                    box = np.array([x1, y1, x2, y2])
                track_copy['box'] = box
                track_copy['box_history'] = track['box_history'] + [box]
                if len(track_copy['box_history']) > 5:
                    track_copy['box_history'].pop(0)
                track_copy['box'] = np.mean(track_copy['box_history'], axis=0).astype(int)
                if label in assigned_poses:
                    track_copy['keypoints'] = assigned_poses[label]['pose']['keypoints']
                else:
                    track_copy.pop('keypoints', None)
                current_tracks.append(track_copy)
            else:
                if track['box_history']:
                    track_copy = track.copy()
                    track_copy['box'] = track['box_history'][-1]
                    current_tracks.append(track_copy)
                continue
        boxes_for_stabilizer = [track_copy['box'] for track_copy in current_tracks if 'box' in track_copy]
        frame_transform = stabilizer.process_frame(frame, boxes_for_stabilizer)
        for track_copy in current_tracks:
            _apply_transform_to_track(track_copy, frame_transform)

        tracks = current_tracks
        tracks_per_frame.append([track.copy() for track in tracks])
        torch.cuda.empty_cache()
    cap.release()

    # Generate and save CSV data
    left_xdata, left_ydata, right_xdata, right_ydata, c, checker_list, video_angle = process_video_and_extract_data(tracks_per_frame, video_path)
    save_data_to_csv(left_xdata, left_ydata, right_xdata, right_ydata, c, checker_list, video_angle, csv_output_dir)

    # Cleanup compressed video
    if compressed_path and os.path.exists(compressed_path):
        try:
            os.remove(compressed_path)
            print(f"Deleted temporary compressed video: {compressed_path}")
        except Exception as e:
            print(f"Warning: Could not delete temporary file {compressed_path}: {e}")

    summary = stabilizer.get_debug_summary()
    logging.info(
        "Camera stabilization summary: frames=%s, successful_tracks=%s, total_motion=(%.2f, %.2f)",
        summary.get('frames_processed'),
        summary.get('successful_tracks'),
        summary.get('total_motion_x', 0.0),
        summary.get('total_motion_y', 0.0)
    )

    return left_xdata, left_ydata, right_xdata, right_ydata, c, checker_list, video_angle, None
