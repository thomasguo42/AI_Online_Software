"""Camera motion compensation utilities for fencing video analysis."""

from __future__ import annotations

import cv2
import logging
import numpy as np
from typing import List, Tuple


logger = logging.getLogger(__name__)


class ImprovedCameraStabilizer:
    """Estimate and compensate camera translation using background features."""

    def __init__(self, use_strip_detection: bool = True) -> None:
        self.orb = cv2.ORB_create(nfeatures=3000)
        try:
            self.sift = cv2.SIFT_create(nfeatures=2000)
        except Exception:  # pragma: no cover - optional dependency
            self.sift = None
            logger.debug("SIFT unavailable, falling back to ORB only")

        self.lk_params = dict(
            winSize=(21, 21),
            maxLevel=3,
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01)
        )

        self.prev_gray: np.ndarray | None = None
        self.cumulative_transform = np.eye(3, dtype=np.float32)
        self.transform_history: List[Tuple[float, float]] = []

        self.use_strip_detection = use_strip_detection
        self.strip_lines: List[np.ndarray] | None = None

        self.debug_info = {
            'frames_processed': 0,
            'successful_tracks': 0,
            'strip_detections': 0
        }

    @staticmethod
    def _create_comprehensive_mask(
        frame_shape: Tuple[int, int],
        fencer_bboxes: List[Tuple[float, float, float, float]]
    ) -> np.ndarray:
        height, width = frame_shape
        mask = np.ones((height, width), dtype=np.uint8) * 255

        for bbox in fencer_bboxes:
            x1, y1, x2, y2 = map(int, bbox)
            w = x2 - x1
            h = y2 - y1
            if w <= 0 or h <= 0:
                continue

            expand_x = int(w * 0.2)
            expand_y = int(h * 0.1)

            x1 = max(0, x1 - expand_x)
            x2 = min(width, x2 + expand_x)
            y1 = max(0, y1 - expand_y)
            y2 = min(height, y2 + expand_y)

            mask[y1:y2, x1:x2] = 0

        return mask

    @staticmethod
    def _detect_strip_lines(frame: np.ndarray, mask: np.ndarray) -> List[np.ndarray]:
        if frame.ndim == 3:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        else:
            gray = frame

        masked_gray = cv2.bitwise_and(gray, gray, mask=mask)
        edges = cv2.Canny(masked_gray, 30, 100, apertureSize=3)
        lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=80, minLineLength=100, maxLineGap=50)

        if lines is None:
            return []

        horizontal_lines = []
        for line in lines:
            x1, y1, x2, y2 = line[0]
            angle = abs(np.arctan2(y2 - y1, x2 - x1) * 180 / np.pi)
            if angle < 30 or angle > 150:
                horizontal_lines.append(line[0])
        return horizontal_lines

    def _track_features_optical_flow(
        self,
        prev_gray: np.ndarray,
        curr_gray: np.ndarray,
        mask: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        corners = cv2.goodFeaturesToTrack(
            prev_gray,
            mask=mask,
            maxCorners=500,
            qualityLevel=0.01,
            minDistance=10,
            blockSize=7
        )

        if corners is None or len(corners) < 10:
            return np.array([]), np.array([])

        next_corners, status, _ = cv2.calcOpticalFlowPyrLK(prev_gray, curr_gray, corners, None, **self.lk_params)

        good_prev = corners[status == 1].reshape(-1, 2)
        good_next = next_corners[status == 1].reshape(-1, 2)
        return good_prev, good_next

    @staticmethod
    def _calculate_motion_from_strips(
        prev_lines: List[np.ndarray],
        curr_lines: List[np.ndarray]
    ) -> Tuple[float, float]:
        if len(prev_lines) < 2 or len(curr_lines) < 2:
            return 0.0, 0.0

        prev_y = [(line[1] + line[3]) / 2 for line in prev_lines]
        curr_y = [(line[1] + line[3]) / 2 for line in curr_lines]

        x_shifts, y_shifts = [], []
        for i, y_prev in enumerate(prev_y):
            best_j = -1
            best_dist = float('inf')
            for j, y_curr in enumerate(curr_y):
                dist = abs(y_curr - y_prev)
                if dist < best_dist and dist < 50:
                    best_dist = dist
                    best_j = j

            if best_j >= 0:
                prev_mid_x = (prev_lines[i][0] + prev_lines[i][2]) / 2
                curr_mid_x = (curr_lines[best_j][0] + curr_lines[best_j][2]) / 2
                x_shifts.append(curr_mid_x - prev_mid_x)
                y_shifts.append(curr_y[best_j] - y_prev)

        if x_shifts:
            return float(np.median(x_shifts)), float(np.median(y_shifts))
        return 0.0, 0.0

    def _estimate_camera_motion(
        self,
        prev_gray: np.ndarray,
        curr_gray: np.ndarray,
        mask: np.ndarray
    ) -> Tuple[float, float]:
        dx_estimates: List[float] = []
        dy_estimates: List[float] = []
        weights: List[float] = []

        prev_pts, next_pts = self._track_features_optical_flow(prev_gray, curr_gray, mask)
        if len(prev_pts) >= 10:
            try:
                transform, inliers = cv2.estimateAffinePartial2D(
                    prev_pts,
                    next_pts,
                    method=cv2.RANSAC,
                    ransacReprojThreshold=3.0,
                    maxIters=1000
                )
                if transform is not None:
                    dx = transform[0, 2]
                    dy = transform[1, 2]
                    inlier_ratio = float(np.sum(inliers) / len(inliers))
                    dx_estimates.append(dx)
                    dy_estimates.append(dy)
                    weights.append(inlier_ratio * 2.0)
            except Exception:  # pragma: no cover - guard against cv2 failures
                pass

        if self.use_strip_detection:
            curr_lines = self._detect_strip_lines(curr_gray, mask)
            if self.strip_lines is not None and curr_lines:
                strip_dx, strip_dy = self._calculate_motion_from_strips(self.strip_lines, curr_lines)
                if abs(strip_dx) > 0.1 or abs(strip_dy) > 0.1:
                    dx_estimates.append(strip_dx)
                    dy_estimates.append(strip_dy)
                    weights.append(1.5)
                    self.debug_info['strip_detections'] += 1
            self.strip_lines = curr_lines

        if not dx_estimates:
            detector = self.sift if self.sift else self.orb
            kp1, des1 = detector.detectAndCompute(prev_gray, mask)
            kp2, des2 = detector.detectAndCompute(curr_gray, mask)

            if des1 is not None and des2 is not None and len(des1) >= 5:
                matcher = cv2.BFMatcher(cv2.NORM_L2 if self.sift else cv2.NORM_HAMMING)
                matches = matcher.knnMatch(des1, des2, k=2)
                good_matches = []
                for pair in matches:
                    if len(pair) == 2:
                        m, n = pair
                        if m.distance < 0.7 * n.distance:
                            good_matches.append(m)

                if len(good_matches) >= 5:
                    src_pts = np.float32([kp1[m.queryIdx].pt for m in good_matches])
                    dst_pts = np.float32([kp2[m.trainIdx].pt for m in good_matches])
                    motion = dst_pts - src_pts
                    dx_estimates.append(float(np.median(motion[:, 0])))
                    dy_estimates.append(float(np.median(motion[:, 1])))
                    weights.append(0.5)

        if dx_estimates:
            weights = np.array(weights, dtype=np.float32)
            weights /= weights.sum()
            final_dx = float(np.sum(np.array(dx_estimates) * weights))
            final_dy = float(np.sum(np.array(dy_estimates) * weights))
            self.debug_info['successful_tracks'] += 1
            return final_dx, final_dy

        return 0.0, 0.0

    def process_frame(
        self,
        frame_bgr: np.ndarray,
        fencer_bboxes: List[Tuple[float, float, float, float]]
    ) -> np.ndarray:
        curr_gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY) if frame_bgr.ndim == 3 else frame_bgr
        self.debug_info['frames_processed'] += 1

        if self.prev_gray is None:
            self.prev_gray = curr_gray.copy()
            return self.cumulative_transform.copy()

        mask = self._create_comprehensive_mask(curr_gray.shape, fencer_bboxes)
        dx, dy = self._estimate_camera_motion(self.prev_gray, curr_gray, mask)

        self.transform_history.append((dx, dy))
        if len(self.transform_history) > 5:
            self.transform_history.pop(0)

        if len(self.transform_history) >= 3:
            weights = np.array([1, 2, 3], dtype=np.float32)[:len(self.transform_history)]
            weights /= weights.sum()
            dx_smooth = float(np.sum([h[0] * w for h, w in zip(self.transform_history, weights)]))
            dy_smooth = float(np.sum([h[1] * w for h, w in zip(self.transform_history, weights)]))
        else:
            dx_smooth, dy_smooth = dx, dy

        transform = np.array(
            [[1, 0, -dx_smooth],
             [0, 1, -dy_smooth],
             [0, 0, 1]],
            dtype=np.float32
        )

        self.cumulative_transform = self.cumulative_transform @ transform
        self.prev_gray = curr_gray.copy()

        if self.debug_info['frames_processed'] % 50 == 0:
            total_dx = float(self.cumulative_transform[0, 2])
            total_dy = float(self.cumulative_transform[1, 2])
            logger.info(
                "Camera stabilization progress: frame %d, cumulative (dx=%.1f, dy=%.1f)",
                self.debug_info['frames_processed'], total_dx, total_dy
            )

        return self.cumulative_transform.copy()

    def get_debug_summary(self) -> dict:
        if self.debug_info['frames_processed'] == 0:
            return {'status': 'No frames processed'}

        return {
            'frames_processed': self.debug_info['frames_processed'],
            'successful_tracks': self.debug_info['successful_tracks'],
            'strip_detections': self.debug_info['strip_detections'],
            'total_motion_x': float(self.cumulative_transform[0, 2]),
            'total_motion_y': float(self.cumulative_transform[1, 2])
        }


def transform_points(points: np.ndarray, transform: np.ndarray) -> np.ndarray:
    """Apply homogeneous transform to Nx2 array of points."""
    if points.size == 0:
        return points

    if points.ndim == 1:
        points = points.reshape(1, -1)

    ones = np.ones((points.shape[0], 1), dtype=np.float32)
    homogenous = np.concatenate([points.astype(np.float32), ones], axis=1)
    transformed = (transform @ homogenous.T).T[:, :2]
    return transformed

