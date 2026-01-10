#!/usr/bin/env python3
"""
Auto-label blade tip keypoints using SAM initialization + XMem tracking.

This script uses a single click per blade on a chosen init frame, tracks the
blade masks over the video, and extracts the blade tip per frame by selecting
the mask endpoint closest to the initial click.
"""

import argparse
import os
import sys

import cv2
import numpy as np
import torch

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(ROOT_DIR)
if PROJECT_DIR not in sys.path:
    sys.path.insert(0, PROJECT_DIR)

from your_scripts.tools.interact_tools import SamControler
from your_scripts.tracker.base_tracker import BaseTracker


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", required=True, help="Path to input video")
    parser.add_argument("--output-dir", required=True, help="Output directory")
    parser.add_argument("--sam-checkpoint", default=os.path.join(YOUR_SCRIPTS_DIR, "checkpoints", "sam_vit_h_4b8939.pth"))
    parser.add_argument("--xmem-checkpoint", default=os.path.join(YOUR_SCRIPTS_DIR, "checkpoints", "XMem-s012.pth"))
    parser.add_argument("--sam-model-type", default="vit_h")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--init-frame", type=int, default=0, help="Frame index for SAM init")
    parser.add_argument("--mask-dilate", type=int, default=5, help="Dilate SAM mask by N pixels")
    parser.add_argument("--left-tip", default=None, help="Left blade tip click as 'x,y'")
    parser.add_argument("--right-tip", default=None, help="Right blade tip click as 'x,y'")
    parser.add_argument("--output-video", default=None, help="Optional overlay video output (mp4)")
    return parser.parse_args()


def _parse_point(text):
    if text is None:
        return None
    parts = text.split(",")
    if len(parts) != 2:
        raise ValueError(f"Invalid point '{text}', expected 'x,y'")
    return np.array([float(parts[0]), float(parts[1])], dtype=np.float32)


def _load_frame(cap, target_idx):
    cap.set(cv2.CAP_PROP_POS_FRAMES, target_idx)
    ret, frame_bgr = cap.read()
    if not ret:
        raise RuntimeError(f"Failed to read frame {target_idx}")
    return cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)


def _pick_points(frame_rgb, num_points):
    points = []
    window_name = "Select Blade Tips (Left then Right)"

    def on_mouse(event, x, y, _flags, _param):
        if event == cv2.EVENT_LBUTTONDOWN:
            points.append((x, y))

    frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(window_name, on_mouse)

    while True:
        display = frame_bgr.copy()
        for idx, (x, y) in enumerate(points):
            color = (255, 0, 0) if idx == 0 else (0, 0, 255)
            cv2.circle(display, (x, y), 6, color, -1)
            cv2.putText(display, f"{idx + 1}", (x + 8, y - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        cv2.putText(display, "Click left tip then right tip. Press Enter to finish.",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.imshow(window_name, display)
        key = cv2.waitKey(1) & 0xFF
        if key in (13, 10) and len(points) >= num_points:
            break
        if key == 27:
            break

    cv2.destroyWindow(window_name)
    if len(points) < num_points:
        raise RuntimeError("Point selection canceled or incomplete.")
    return [np.array(p, dtype=np.float32) for p in points[:num_points]]


def _build_template_mask(frame_rgb, sam_controler, blade_points, mask_dilate):
    template_mask = np.zeros(frame_rgb.shape[:2], dtype=np.uint8)
    kernel = None
    if mask_dilate > 0:
        kernel = np.ones((mask_dilate, mask_dilate), np.uint8)

    for blade_id, point in blade_points.items():
        points = np.array([[int(point[0]), int(point[1])]], dtype=np.int32)
        labels = np.array([1], dtype=np.int32)

        sam_controler.sam_controler.reset_image()
        sam_controler.sam_controler.set_image(frame_rgb)
        mask, _, _ = sam_controler.first_frame_click(
            image=frame_rgb,
            points=points,
            labels=labels,
            multimask=True,
        )

        binary_mask = (mask > 0).astype(np.uint8)
        if kernel is not None:
            binary_mask = cv2.dilate(binary_mask, kernel, iterations=1)
        if binary_mask.max() == 0:
            raise RuntimeError(f"SAM produced empty mask for blade {blade_id}")

        template_mask[binary_mask == 1] = blade_id
    return template_mask


def _mask_tip(mask, tip_hint):
    ys, xs = np.where(mask > 0)
    if len(xs) < 2:
        return None
    coords = np.stack([xs, ys], axis=1).astype(np.float32)
    mean = coords.mean(axis=0)
    centered = coords - mean
    cov = np.cov(centered.T)
    eigvals, eigvecs = np.linalg.eigh(cov)
    axis = eigvecs[:, np.argmax(eigvals)]
    proj = centered @ axis
    p1 = coords[np.argmin(proj)]
    p2 = coords[np.argmax(proj)]
    d1 = np.linalg.norm(p1 - tip_hint)
    d2 = np.linalg.norm(p2 - tip_hint)
    return p1 if d1 <= d2 else p2


def main():
    args = parse_args()
    left_tip = _parse_point(args.left_tip)
    right_tip = _parse_point(args.right_tip)

    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        raise FileNotFoundError(f"Could not open video {args.video}")

    device = args.device if torch.cuda.is_available() else "cpu"
    sam_controler = SamControler(args.sam_checkpoint, args.sam_model_type, device)
    tracker = BaseTracker(args.xmem_checkpoint, device)

    init_frame_rgb = _load_frame(cap, args.init_frame)
    if left_tip is None or right_tip is None:
        left_tip, right_tip = _pick_points(init_frame_rgb, 2)
    blade_points = {1: left_tip, 2: right_tip}
    template_mask = _build_template_mask(
        init_frame_rgb, sam_controler, blade_points, args.mask_dilate
    )

    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    frame_idx = 0
    tip_refs = {1: left_tip, 2: right_tip}
    prev_tips = {1: None, 2: None}
    rows = []

    writer = None
    if args.output_video:
        fps = cap.get(cv2.CAP_PROP_FPS) or 30
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        os.makedirs(os.path.dirname(args.output_video) or ".", exist_ok=True)
        writer = cv2.VideoWriter(
            args.output_video,
            cv2.VideoWriter_fourcc(*"mp4v"),
            fps,
            (width, height),
        )

    while True:
        ret, frame_bgr = cap.read()
        if not ret:
            break
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        if frame_idx == 0:
            mask, _, _ = tracker.track(frame_rgb, template_mask)
        else:
            mask, _, _ = tracker.track(frame_rgb)

        for blade_id in (1, 2):
            blade_mask = (mask == blade_id).astype(np.uint8)
            tip_hint = prev_tips[blade_id] if prev_tips[blade_id] is not None else tip_refs[blade_id]
            tip = _mask_tip(blade_mask, tip_hint)
            if tip is None:
                rows.append([frame_idx, blade_id, "", "", 0])
                continue
            prev_tips[blade_id] = tip
            rows.append([
                frame_idx,
                blade_id,
                float(tip[0]),
                float(tip[1]),
                int(blade_mask.sum()),
            ])

        if writer is not None:
            overlay = frame_bgr.copy()
            for blade_id, color in ((1, (255, 0, 0)), (2, (0, 0, 255))):
                blade_mask = (mask == blade_id).astype(np.uint8)
                if blade_mask.max() > 0:
                    colored = np.zeros_like(overlay, dtype=np.uint8)
                    colored[blade_mask == 1] = color
                    overlay = cv2.addWeighted(overlay, 1.0, colored, 0.35, 0)
            for blade_id, color in ((1, (255, 0, 0)), (2, (0, 0, 255))):
                tip = prev_tips[blade_id]
                if tip is not None:
                    cv2.circle(overlay, (int(tip[0]), int(tip[1])), 6, color, -1)
                    cv2.putText(overlay, f"{blade_id}", (int(tip[0]) + 8, int(tip[1]) - 8),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            writer.write(overlay)

        frame_idx += 1

    cap.release()
    if writer is not None:
        writer.release()

    os.makedirs(args.output_dir, exist_ok=True)
    csv_path = os.path.join(args.output_dir, "blade_tip_keypoints.csv")
    with open(csv_path, "w", encoding="utf-8") as f:
        f.write("frame,blade_id,tip_x,tip_y,mask_area\n")
        for row in rows:
            f.write(",".join(map(str, row)) + "\n")


if __name__ == "__main__":
    main()
