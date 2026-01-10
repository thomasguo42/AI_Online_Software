#!/usr/bin/env python3
"""
Generate a video with SAM mask overlays and fencing keypoints.

This script combines:
  - SAM + XMem tracking (from Track-Anything) to produce segmentation masks
  - Keypoint overlays from the fencer tracking CSVs

Inputs:
  --video: path to input video
  --csv-dir: directory containing left_xdata.csv, left_ydata.csv, right_xdata.csv,
             right_ydata.csv, meta.csv
  --output: output video path (mp4)
  --sam-checkpoint: SAM checkpoint path
  --xmem-checkpoint: XMem checkpoint path
"""

import argparse
import os
import sys

import cv2
import numpy as np
import torch
from ultralytics import YOLO

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.dirname(ROOT_DIR)
for path in (ROOT_DIR, PARENT_DIR):
    if path not in sys.path:
        sys.path.insert(0, path)

TRACK_ANYTHING_DIR = os.path.join(PARENT_DIR, "your_scripts")
if TRACK_ANYTHING_DIR not in sys.path:
    sys.path.insert(0, TRACK_ANYTHING_DIR)

from standalone_match_separation import load_data_from_csv

# Ensure we can import Track-Anything modules.
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "your_scripts")))

from your_scripts.tools.interact_tools import SamControler
from your_scripts.tools.painter import mask_painter
from your_scripts.tracker.base_tracker import BaseTracker


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", required=True, help="Path to input video")
    parser.add_argument("--csv-dir", default=None, help="CSV directory with keypoint data")
    parser.add_argument("--output", required=True, help="Output video path (mp4)")
    parser.add_argument("--sam-checkpoint", default=None, help="SAM checkpoint path")
    parser.add_argument("--xmem-checkpoint", default=os.path.join("your_scripts", "checkpoints", "XMem-s012.pth"),
                        help="XMem checkpoint path")
    parser.add_argument("--sam-model-type", default="vit_h", help="SAM model type")
    parser.add_argument("--device", default="cuda:0", help="Device for SAM/XMem")
    parser.add_argument("--yolo-model", default=None, help="YOLO model path/name")
    parser.add_argument("--pose-model", default=None, help="YOLO pose model path/name")
    parser.add_argument("--reid-model", default=os.path.join(
        "your_scripts",
        "checkpoints",
        "osnet_x0_25_msmt17_combineall_256x128_amsgrad_ep150_stp60_lr0.0015_b64_fb10_softmax_labelsmooth_flip_jitter.pth",
    ), help="ReID model path")
    parser.add_argument("--max-people", type=int, default=2, help="Max people to track")
    parser.add_argument("--conf", type=float, default=0.25, help="YOLO confidence threshold")
    parser.add_argument("--yolo-csv-conf", type=float, default=0.7, help="YOLO conf for CSV extraction")
    parser.add_argument("--mask-dilate", type=int, default=9, help="Dilate SAM mask by N pixels")
    return parser.parse_args()


def draw_keypoints(frame_rgb, frame_idx, left_x_df, left_y_df, right_x_df, right_y_df, c):
    if frame_idx >= len(left_x_df):
        return frame_rgb

    for kp in range(7, 17):
        try:
            left_x = left_x_df.loc[frame_idx, str(kp)]
            left_y = left_y_df.loc[frame_idx, str(kp)]
            right_x = right_x_df.loc[frame_idx, str(kp)]
            right_y = right_y_df.loc[frame_idx, str(kp)]
        except KeyError:
            continue

        if not np.isnan(left_x) and not np.isnan(left_y):
            kp_x = int(left_x * c)
            kp_y = int(left_y * c)
            cv2.circle(frame_rgb, (kp_x, kp_y), 4, (255, 0, 0), -1)
            cv2.putText(frame_rgb, f"{left_x:.2f}", (kp_x + 8, kp_y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 0, 0), 1)

        if not np.isnan(right_x) and not np.isnan(right_y):
            kp_x = int(right_x * c)
            kp_y = int(right_y * c)
            cv2.circle(frame_rgb, (kp_x, kp_y), 4, (0, 0, 255), -1)
            cv2.putText(frame_rgb, f"{right_x:.2f}", (kp_x + 8, kp_y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)

    return frame_rgb


def build_template_mask(frame_rgb, sam_controler, yolo_model, max_people, conf_threshold):
    results = yolo_model(frame_rgb, classes=[0], conf=conf_threshold)
    detections = results[0].boxes
    if len(detections) == 0:
        raise RuntimeError("No people detected in the first frame.")

    template_mask = np.zeros(frame_rgb.shape[:2], dtype=np.uint8)
    label_counter = 1

    for det_idx, det in enumerate(detections):
        if det_idx >= max_people:
            break
        box = det.xyxy[0].cpu().numpy()
        center_x = int(np.round((box[0] + box[2]) / 2))
        center_y = int(np.round((box[1] + box[3]) / 2))
        points = np.array([[center_x, center_y]])
        labels = np.array([1])

        sam_controler.sam_controler.reset_image()
        sam_controler.sam_controler.set_image(frame_rgb)
        mask, _, _ = sam_controler.first_frame_click(
            image=frame_rgb,
            points=points,
            labels=labels,
            multimask=True,
        )

        binary_mask = (mask > 0).astype(np.uint8)
        if binary_mask.max() == 0:
            continue
        template_mask[binary_mask == 1] = label_counter
        label_counter += 1

    if template_mask.max() == 0:
        raise RuntimeError("SAM produced empty masks on the first frame.")
    return template_mask


def select_top_people(video_path, yolo_model, conf_threshold, max_people):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Could not open video {video_path}")
    ret, frame_bgr = cap.read()
    cap.release()
    if not ret:
        raise RuntimeError("Failed to read the first frame for selection.")
    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    results = yolo_model(frame_rgb, classes=[0], conf=conf_threshold)
    detections = []
    for idx, det in enumerate(results[0].boxes):
        box = det.xyxy[0].cpu().numpy()
        area = max(0.0, (box[2] - box[0]) * (box[3] - box[1]))
        detections.append((idx, area))
    if len(detections) < max_people:
        raise RuntimeError("Not enough person detections to select two fencers.")
    detections.sort(key=lambda x: x[1], reverse=True)
    return [idx for idx, _ in detections[:max_people]]


def main():
    args = parse_args()

    if args.sam_checkpoint is None:
        local_sam = os.path.join(os.path.dirname(args.video), "sam_vit_h_4b8939.pth")
        args.sam_checkpoint = local_sam if os.path.exists(local_sam) else os.path.join(
            "your_scripts", "checkpoints", "sam_vit_h_4b8939.pth"
        )

    if args.yolo_model is None:
        local_yolo = os.path.join(os.path.dirname(args.video), "yolov8x.pt")
        args.yolo_model = local_yolo if os.path.exists(local_yolo) else os.path.join(
            "your_scripts", "yolov8x.pt"
        )

    if args.pose_model is None:
        local_pose = os.path.join(os.path.dirname(args.video), "yolov8x-pose.pt")
        args.pose_model = local_pose if os.path.exists(local_pose) else os.path.join(
            "your_scripts", "yolov8x-pose.pt"
        )

    csv_dir = args.csv_dir
    if csv_dir is None:
        csv_dir = os.path.join(os.path.dirname(args.video), "csv_output")
        os.makedirs(csv_dir, exist_ok=True)
        yolo_for_csv = YOLO(args.yolo_model)
        selected_indexes = select_top_people(args.video, yolo_for_csv, args.yolo_csv_conf, args.max_people)
        from your_scripts import video_analysis
        video_analysis.main(
            args.video,
            os.path.join(csv_dir, "keypoints_debug.mp4"),
            args.reid_model,
            csv_dir,
            selected_indexes,
            fps=30,
            yolo_model_path=args.yolo_model,
            pose_model_path=args.pose_model,
            sam_checkpoint=args.sam_checkpoint,
            sam_model_type=args.sam_model_type,
            xmem_checkpoint=args.xmem_checkpoint,
            yolo_conf=args.yolo_csv_conf,
            enable_stabilization=False,
        )

    left_x_df, left_y_df, right_x_df, right_y_df, c, _, _ = load_data_from_csv(csv_dir)

    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        raise FileNotFoundError(f"Could not open video {args.video}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    writer = cv2.VideoWriter(
        args.output,
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height),
    )

    device = args.device if torch.cuda.is_available() else "cpu"
    sam_controler = SamControler(args.sam_checkpoint, args.sam_model_type, device)
    tracker = BaseTracker(args.xmem_checkpoint, device)
    yolo_model = YOLO(args.yolo_model)

    ret, frame_bgr = cap.read()
    if not ret:
        raise RuntimeError("Failed to read the first frame from video.")

    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    template_mask = build_template_mask(
        frame_rgb,
        sam_controler,
        yolo_model,
        args.max_people,
        args.conf,
    )

    frame_idx = 0
    kernel = None
    if args.mask_dilate > 0:
        kernel = np.ones((args.mask_dilate, args.mask_dilate), np.uint8)

    while ret:
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        if frame_idx == 0:
            mask, _, painted = tracker.track(frame_rgb, template_mask)
        else:
            mask, _, painted = tracker.track(frame_rgb)

        # Repaint with optional dilation to enlarge the visible mesh.
        painted = frame_rgb.copy()
        num_objs = int(mask.max())
        for obj_id in range(1, num_objs + 1):
            obj_mask = (mask == obj_id).astype("uint8")
            if kernel is not None:
                obj_mask = cv2.dilate(obj_mask, kernel, iterations=1)
            painted = mask_painter(painted, obj_mask, mask_color=obj_id + 1)

        painted = draw_keypoints(
            painted,
            frame_idx,
            left_x_df,
            left_y_df,
            right_x_df,
            right_y_df,
            c,
        )

        writer.write(cv2.cvtColor(painted, cv2.COLOR_RGB2BGR))
        ret, frame_bgr = cap.read()
        frame_idx += 1

    tracker.clear_memory()
    cap.release()
    writer.release()


if __name__ == "__main__":
    main()
