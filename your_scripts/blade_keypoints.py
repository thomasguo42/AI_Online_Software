#!/usr/bin/env python3
"""
Extract blade tip/base keypoints from a video using SAM for initialization and XMem for tracking.

Usage example:
  python blade_keypoints.py --video /path/to/video.mp4 --output-csv /path/to/blade_keypoints.csv \
    --blade-points '[{"id":1,"points":[[320,220]]},{"id":2,"points":[[960,240]]}]'
"""

import argparse
import json
import os
import sys

import cv2
import numpy as np
import torch

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from tools.interact_tools import SamControler
from tracker.base_tracker import BaseTracker


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", required=True, help="Path to input video")
    parser.add_argument("--output-csv", required=True, help="Output CSV path")
    parser.add_argument("--sam-checkpoint", default=os.path.join("checkpoints", "sam_vit_h_4b8939.pth"))
    parser.add_argument("--xmem-checkpoint", default=os.path.join("checkpoints", "XMem-s012.pth"))
    parser.add_argument("--sam-model-type", default="vit_h")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--frame-index", type=int, default=0, help="Frame index for SAM init")
    parser.add_argument("--mask-dilate", type=int, default=5, help="Dilate SAM mask by N pixels")
    parser.add_argument(
        "--blade-points",
        required=True,
        help="JSON list of blade specs: [{'id':1,'points':[[x,y],...]}, ...]",
    )
    return parser.parse_args()


def _load_frame(cap, target_idx):
    cap.set(cv2.CAP_PROP_POS_FRAMES, target_idx)
    ret, frame_bgr = cap.read()
    if not ret:
        raise RuntimeError(f"Failed to read frame {target_idx}")
    return cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)


def _build_template_mask(frame_rgb, sam_controler, blade_specs, mask_dilate):
    template_mask = np.zeros(frame_rgb.shape[:2], dtype=np.uint8)
    kernel = None
    if mask_dilate > 0:
        kernel = np.ones((mask_dilate, mask_dilate), np.uint8)

    for blade in blade_specs:
        blade_id = int(blade["id"])
        points = np.array(blade["points"], dtype=np.int32)
        labels = np.ones((len(points),), dtype=np.int32)

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


def _mask_endpoints(mask, tip_hint=None):
    ys, xs = np.where(mask > 0)
    if len(xs) < 2:
        return None, None
    coords = np.stack([xs, ys], axis=1).astype(np.float32)
    mean = coords.mean(axis=0)
    centered = coords - mean
    cov = np.cov(centered.T)
    eigvals, eigvecs = np.linalg.eigh(cov)
    axis = eigvecs[:, np.argmax(eigvals)]
    proj = centered @ axis
    p1 = coords[np.argmin(proj)]
    p2 = coords[np.argmax(proj)]
    if tip_hint is None:
        return p1, p2
    d1 = np.linalg.norm(p1 - tip_hint)
    d2 = np.linalg.norm(p2 - tip_hint)
    if d1 <= d2:
        return p1, p2
    return p2, p1


def main():
    args = parse_args()

    blade_specs = json.loads(args.blade_points)
    if not isinstance(blade_specs, list) or not blade_specs:
        raise ValueError("blade-points must be a non-empty list of blade specs")

    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        raise FileNotFoundError(f"Could not open video {args.video}")

    device = args.device if torch.cuda.is_available() else "cpu"
    sam_controler = SamControler(args.sam_checkpoint, args.sam_model_type, device)
    tracker = BaseTracker(args.xmem_checkpoint, device)

    init_frame_rgb = _load_frame(cap, args.frame_index)
    template_mask = _build_template_mask(
        init_frame_rgb, sam_controler, blade_specs, args.mask_dilate
    )

    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    frame_idx = 0
    tip_refs = {}
    prev_tips = {}
    for blade in blade_specs:
        blade_id = int(blade["id"])
        tip_refs[blade_id] = np.array(blade["points"][0], dtype=np.float32)
        prev_tips[blade_id] = None

    rows = []
    while True:
        ret, frame_bgr = cap.read()
        if not ret:
            break
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        if frame_idx == 0:
            mask, _, _ = tracker.track(frame_rgb, template_mask)
        else:
            mask, _, _ = tracker.track(frame_rgb)

        for blade in blade_specs:
            blade_id = int(blade["id"])
            blade_mask = (mask == blade_id).astype(np.uint8)
            tip_hint = prev_tips[blade_id] if prev_tips[blade_id] is not None else tip_refs[blade_id]
            tip, base = _mask_endpoints(blade_mask, tip_hint=tip_hint)
            if tip is None or base is None:
                rows.append([frame_idx, blade_id, "", "", "", "", 0])
                continue
            prev_tips[blade_id] = tip
            rows.append([
                frame_idx,
                blade_id,
                float(tip[0]),
                float(tip[1]),
                float(base[0]),
                float(base[1]),
                int(blade_mask.sum()),
            ])
        frame_idx += 1

    cap.release()

    os.makedirs(os.path.dirname(args.output_csv) or ".", exist_ok=True)
    with open(args.output_csv, "w", encoding="utf-8") as f:
        f.write("frame,blade_id,tip_x,tip_y,base_x,base_y,mask_area\n")
        for row in rows:
            f.write(",".join(map(str, row)) + "\n")


if __name__ == "__main__":
    main()
