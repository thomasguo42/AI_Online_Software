import cv2
import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Tuple, List, Optional
from ultralytics import YOLO
import logging
import os
import math

# Use Agg backend for headless Matplotlib rendering
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Constants
POSE_KEYPOINT_NAMES = [
    'nose', 'left_eye', 'right_eye', 'left_ear', 'right_ear',
    'left_shoulder', 'right_shoulder', 'left_elbow', 'right_elbow',
    'left_wrist', 'right_wrist', 'left_hip', 'right_hip',
    'left_knee', 'right_knee', 'left_ankle', 'right_ankle'
]

# COCO skeleton edges by keypoint indices (using your 17-keypoint order)
COCO_EDGES = [
    (0,1),(0,2),(1,3),(2,4),          # face
    (5,7),(7,9), (6,8),(8,10),        # arms
    (5,6),(5,11),(6,12),              # torso top
    (11,12),(11,13),(13,15),          # left leg
    (12,14),(14,16)                   # right leg
]

@dataclass
class FencerData:
    """A structure to hold all data for a single detected fencer in a frame."""
    keypoints: np.ndarray  # Shape: (17, 2) for (x, y) coordinates
    confidence: np.ndarray  # Shape: (17,) for confidence scores
    bbox: Tuple[float, float, float, float]  # Bounding box (x1, y1, x2, y2)

class ImprovedCameraStabilizer:
    """Enhanced camera stabilizer specifically designed for fencing videos."""

    def __init__(self, use_strip_detection: bool = True):
        """Initialize the improved stabilizer."""
        # Feature detectors
        self.orb = cv2.ORB_create(nfeatures=3000)
        self.sift = None
        try:
            self.sift = cv2.SIFT_create(nfeatures=2000)
        except Exception:
            logger.warning("SIFT not available, using ORB only")

        # Optical flow parameters
        self.lk_params = dict(
            winSize=(21, 21),
            maxLevel=3,
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01)
        )

        # State
        self.prev_gray = None
        self.prev_features = None
        self.cumulative_transform = np.eye(3, dtype=np.float32)
        self.transform_history = []

        # Strip detection
        self.use_strip_detection = use_strip_detection
        self.strip_lines = None

        # Debug
        self.debug_info = {
            'frames_processed': 0,
            'successful_tracks': 0,
            'strip_detections': 0
        }

    def _create_comprehensive_mask(self, frame_shape: Tuple[int, int],
                                  fencer_bboxes: List[Tuple[float, float, float, float]]) -> np.ndarray:
        """Create a mask that properly excludes fencer regions."""
        height, width = frame_shape
        mask = np.ones((height, width), dtype=np.uint8) * 255

        for bbox in fencer_bboxes:
            x1, y1, x2, y2 = map(int, bbox)

            # Expand bbox by 20% to ensure we exclude all fencer pixels
            w = x2 - x1
            h = y2 - y1
            expand_x = int(w * 0.2)
            expand_y = int(h * 0.1)  # Less vertical expansion

            x1 = max(0, x1 - expand_x)
            x2 = min(width, x2 + expand_x)
            y1 = max(0, y1 - expand_y)
            y2 = min(height, y2 + expand_y)

            # Mask out the entire expanded region
            mask[y1:y2, x1:x2] = 0

        return mask

    def _detect_strip_lines(self, frame: np.ndarray, mask: np.ndarray) -> List[np.ndarray]:
        """Detect fencing strip lines using Hough transform."""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if len(frame.shape) == 3 else frame

        # Apply mask to focus on background
        masked_gray = cv2.bitwise_and(gray, gray, mask=mask)

        # Edge detection with Canny
        edges = cv2.Canny(masked_gray, 30, 100, apertureSize=3)

        # Detect lines using HoughLinesP
        lines = cv2.HoughLinesP(
            edges,
            rho=1,
            theta=np.pi/180,
            threshold=80,
            minLineLength=100,
            maxLineGap=50
        )

        if lines is None:
            return []

        # Filter for horizontal-ish lines (strip boundaries)
        horizontal_lines = []
        for line in lines:
            x1, y1, x2, y2 = line[0]
            angle = abs(np.arctan2(y2 - y1, x2 - x1) * 180 / np.pi)
            # Keep lines within 30 degrees of horizontal
            if angle < 30 or angle > 150:
                horizontal_lines.append(line[0])

        return horizontal_lines

    def _track_features_optical_flow(self, prev_gray: np.ndarray, curr_gray: np.ndarray,
                                    mask: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Use optical flow for more robust feature tracking."""
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

        # Track features using Lucas-Kanade optical flow
        next_corners, status, _ = cv2.calcOpticalFlowPyrLK(
            prev_gray, curr_gray, corners, None, **self.lk_params
        )

        # Keep only successfully tracked points
        good_prev = corners[status == 1].reshape(-1, 2)
        good_next = next_corners[status == 1].reshape(-1, 2)

        return good_prev, good_next

    def _calculate_motion_from_strips(self, prev_lines: List[np.ndarray],
                                     curr_lines: List[np.ndarray]) -> Tuple[float, float]:
        """Calculate camera motion by tracking strip line movement."""
        if len(prev_lines) < 2 or len(curr_lines) < 2:
            return 0.0, 0.0

        # Match lines between frames based on y-position
        prev_y_positions = [(line[1] + line[3]) / 2 for line in prev_lines]
        curr_y_positions = [(line[1] + line[3]) / 2 for line in curr_lines]

        y_shifts, x_shifts = [], []

        for i, prev_y in enumerate(prev_y_positions):
            min_dist = float('inf')
            best_match = -1
            for j, curr_y in enumerate(curr_y_positions):
                dist = abs(curr_y - prev_y)
                if dist < min_dist and dist < 50:
                    min_dist = dist
                    best_match = j

            if best_match >= 0:
                prev_mid_x = (prev_lines[i][0] + prev_lines[i][2]) / 2
                curr_mid_x = (curr_lines[best_match][0] + curr_lines[best_match][2]) / 2
                x_shifts.append(curr_mid_x - prev_mid_x)
                y_shifts.append(curr_y_positions[best_match] - prev_y)

        if len(x_shifts) > 0:
            dx = np.median(x_shifts)
            dy = np.median(y_shifts)
            return float(dx), float(dy)

        return 0.0, 0.0

    def _estimate_camera_motion(self, prev_gray: np.ndarray, curr_gray: np.ndarray,
                               mask: np.ndarray) -> Tuple[float, float]:
        """Estimate camera motion using multiple methods."""
        dx_estimates, dy_estimates, weights = [], [], []

        # Method 1: Optical flow tracking
        prev_pts, next_pts = self._track_features_optical_flow(prev_gray, curr_gray, mask)
        if len(prev_pts) >= 10:
            try:
                transform, inliers = cv2.estimateAffinePartial2D(
                    prev_pts, next_pts,
                    method=cv2.RANSAC,
                    ransacReprojThreshold=3.0,
                    maxIters=1000
                )
                if transform is not None:
                    optical_dx = transform[0, 2]
                    optical_dy = transform[1, 2]
                    inlier_ratio = float(np.sum(inliers) / len(inliers))

                    dx_estimates.append(optical_dx)
                    dy_estimates.append(optical_dy)
                    weights.append(inlier_ratio * 2.0)
                    logger.debug(f"Optical flow: dx={optical_dx:.2f}, dy={optical_dy:.2f}, inliers={inlier_ratio:.2%}")
            except Exception:
                pass

        # Method 2: Strip line tracking
        if self.use_strip_detection:
            curr_lines = self._detect_strip_lines(curr_gray, mask)
            if self.strip_lines is not None and len(curr_lines) > 0:
                strip_dx, strip_dy = self._calculate_motion_from_strips(self.strip_lines, curr_lines)
                if abs(strip_dx) > 0.1 or abs(strip_dy) > 0.1:
                    dx_estimates.append(strip_dx)
                    dy_estimates.append(strip_dy)
                    weights.append(1.5)
                    logger.debug(f"Strip tracking: dx={strip_dx:.2f}, dy={strip_dy:.2f}")
                    self.debug_info['strip_detections'] += 1
            self.strip_lines = curr_lines

        # Method 3: Feature matching fallback
        if len(dx_estimates) == 0:
            detector = self.sift if self.sift else self.orb
            kp1, des1 = detector.detectAndCompute(prev_gray, mask)
            kp2, des2 = detector.detectAndCompute(curr_gray, mask)

            if des1 is not None and des2 is not None and len(des1) >= 5:
                matcher = cv2.BFMatcher(cv2.NORM_L2 if self.sift else cv2.NORM_HAMMING)
                matches = matcher.knnMatch(des1, des2, k=2)

                good_matches = []
                for mp in matches:
                    if len(mp) == 2:
                        m, n = mp
                        if m.distance < 0.7 * n.distance:
                            good_matches.append(m)

                if len(good_matches) >= 5:
                    src_pts = np.float32([kp1[m.queryIdx].pt for m in good_matches])
                    dst_pts = np.float32([kp2[m.trainIdx].pt for m in good_matches])
                    motion = dst_pts - src_pts
                    feature_dx = np.median(motion[:, 0])
                    feature_dy = np.median(motion[:, 1])

                    dx_estimates.append(feature_dx)
                    dy_estimates.append(feature_dy)
                    weights.append(0.5)
                    logger.debug(f"Feature match: dx={feature_dx:.2f}, dy={feature_dy:.2f}")

        # Combine estimates with weighted average
        if len(dx_estimates) > 0:
            weights = np.array(weights, dtype=np.float32)
            weights /= np.sum(weights)
            final_dx = float(np.sum(np.array(dx_estimates) * weights))
            final_dy = float(np.sum(np.array(dy_estimates) * weights))

            self.debug_info['successful_tracks'] += 1
            return final_dx, final_dy

        return 0.0, 0.0

    def process_frame(self, frame: np.ndarray,
                     fencer_bboxes: List[Tuple[float, float, float, float]]) -> np.ndarray:
        """Process a frame and return cumulative stabilization transform."""
        curr_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if len(frame.shape) == 3 else frame
        self.debug_info['frames_processed'] += 1

        if self.prev_gray is None:
            self.prev_gray = curr_gray.copy()
            return self.cumulative_transform.copy()

        mask = self._create_comprehensive_mask(curr_gray.shape, fencer_bboxes)
        dx, dy = self._estimate_camera_motion(self.prev_gray, curr_gray, mask)

        # Smooth short history
        self.transform_history.append((dx, dy))
        if len(self.transform_history) > 5:
            self.transform_history.pop(0)

        if len(self.transform_history) >= 3:
            weights = np.array([1, 2, 3])[:len(self.transform_history)]
            weights = weights / weights.sum()
            dx_smooth = float(np.sum([h[0] * w for h, w in zip(self.transform_history, weights)]))
            dy_smooth = float(np.sum([h[1] * w for h, w in zip(self.transform_history, weights)]))
        else:
            dx_smooth, dy_smooth = dx, dy

        # IMPORTANT: translate by NEGATIVE of estimated motion to compensate
        T = np.array([
            [1, 0, -dx_smooth],
            [0, 1, -dy_smooth],
            [0, 0, 1]
        ], dtype=np.float32)

        # Update cumulative transform
        self.cumulative_transform = self.cumulative_transform @ T

        # Update state
        self.prev_gray = curr_gray.copy()

        # Log every 20 frames
        if self.debug_info['frames_processed'] % 20 == 0:
            total_dx = float(self.cumulative_transform[0, 2])
            total_dy = float(self.cumulative_transform[1, 2])
            logger.info(f"Frame {self.debug_info['frames_processed']}: "
                        f"Cumulative motion: ({total_dx:.1f}, {total_dy:.1f})")

        return self.cumulative_transform.copy()

    def get_debug_summary(self):
        """Return debug statistics."""
        if self.debug_info['frames_processed'] == 0:
            return {'status': 'No frames processed'}

        success_rate = (self.debug_info['successful_tracks'] /
                        self.debug_info['frames_processed'] * 100.0)

        return {
            'frames_processed': self.debug_info['frames_processed'],
            'successful_tracks': self.debug_info['successful_tracks'],
            'strip_detections': self.debug_info['strip_detections'],
            'success_rate': success_rate,
            'total_motion_x': float(self.cumulative_transform[0, 2]),
            'total_motion_y': float(self.cumulative_transform[1, 2])
        }


class FencingVideoAnalyzer:
    """Main analyzer using the improved stabilizer."""

    def __init__(self, yolo_model_path: str = "yolov8n-pose.pt"):
        """Initialize with YOLO model."""
        self.yolo_model = YOLO(yolo_model_path)
        self.stabilizer = ImprovedCameraStabilizer(use_strip_detection=True)

        # Viz state
        self.fig = None
        self.ax = None
        self.canvas = None
        self.left_scatter = None
        self.right_scatter = None
        self.left_lines = []
        self.right_lines = []
        self.info_text = None

    def _run_yolo_inference(self, frame: np.ndarray) -> List[FencerData]:
        """Run YOLO pose estimation on a frame."""
        results = self.yolo_model(frame, verbose=False)
        detected_fencers = []

        for result in results:
            if result.keypoints is not None and result.boxes is not None:
                keypoints_data = result.keypoints.data
                boxes_data = result.boxes.xyxy.data

                for i in range(keypoints_data.shape[0]):
                    kpts = keypoints_data[i, :, :2].cpu().numpy()
                    conf = keypoints_data[i, :, 2].cpu().numpy()
                    bbox = boxes_data[i].cpu().numpy().tolist()

                    if float(np.nanmean(conf)) > 0.3:
                        detected_fencers.append(FencerData(
                            keypoints=kpts,
                            confidence=conf,
                            bbox=tuple(bbox)
                        ))

        return detected_fencers

    def _transform_keypoints(self, fencers: List[FencerData],
                           transform: np.ndarray) -> List[FencerData]:
        """Apply transformation to stabilize keypoints."""
        stabilized = []
        for fencer in fencers:
            # Transform keypoints
            kpts_h = np.hstack([fencer.keypoints, np.ones((17, 1), dtype=np.float32)])
            kpts_stabilized = (transform @ kpts_h.T).T[:, :2]

            # Transform bbox corners
            bbox_corners = np.array([
                [fencer.bbox[0], fencer.bbox[1], 1.0],
                [fencer.bbox[2], fencer.bbox[3], 1.0]
            ], dtype=np.float32)
            bbox_transformed = (transform @ bbox_corners.T).T[:, :2]

            stabilized.append(FencerData(
                keypoints=kpts_stabilized.astype(np.float32),
                confidence=fencer.confidence,
                bbox=(float(bbox_transformed[0, 0]),
                      float(bbox_transformed[0, 1]),
                      float(bbox_transformed[1, 0]),
                      float(bbox_transformed[1, 1]))
            ))

        return stabilized

    # =========================
    # Matplotlib Visualization
    # =========================
    def _init_visualizer(self, width: int, height: int):
        """Initialize a persistent Matplotlib figure sized to the video frame."""
        if self.fig is not None:
            return

        dpi = 100  # controls pixel density of the canvas
        fig_w = max(4, int(math.ceil(width / dpi)))
        fig_h = max(3, int(math.ceil(height / dpi)))

        self.fig, self.ax = plt.subplots(figsize=(fig_w, fig_h), dpi=dpi)
        self.canvas = FigureCanvasAgg(self.fig)
        self.ax.set_xlim(0, width)
        self.ax.set_ylim(height, 0)   # invert y to match image coordinates
        self.ax.set_aspect('equal')
        self.ax.set_title("Stabilized Keypoints Preview")
        self.ax.set_xticks([])
        self.ax.set_yticks([])

        # Pre-create scatters
        self.left_scatter = self.ax.scatter([], [], s=18)   # default color
        self.right_scatter = self.ax.scatter([], [], s=18)  # default color (different via mpl cycle)

        # Pre-create line artists for skeletons
        for _ in COCO_EDGES:
            l_line, = self.ax.plot([], [], linewidth=2)
            r_line, = self.ax.plot([], [], linewidth=2)
            self.left_lines.append(l_line)
            self.right_lines.append(r_line)

        self.info_text = self.ax.text(10, 20, "", fontsize=10, va='top')

        self.fig.tight_layout()

    def _update_artist_for_fencer(self, fencer: Optional[FencerData],
                                  scatter_artist,
                                  line_artists: List[plt.Line2D]):
        """Update scatter and skeleton lines for a single fencer."""
        if fencer is None or fencer.keypoints is None:
            # Clear artists
            scatter_artist.set_offsets(np.empty((0, 2)))
            for line in line_artists:
                line.set_data([], [])
            return

        kpts = fencer.keypoints
        # Mask invalid points
        valid = ~np.any(np.isnan(kpts), axis=1)
        scatter_artist.set_offsets(kpts[valid, :])

        # Update edges
        for (edge, line) in zip(COCO_EDGES, line_artists):
            i, j = edge
            if valid[i] and valid[j]:
                x = [kpts[i, 0], kpts[j, 0]]
                y = [kpts[i, 1], kpts[j, 1]]
                line.set_data(x, y)
            else:
                line.set_data([], [])

    def _render_viz_frame(self,
                          frame_idx: int,
                          frame_w: int,
                          frame_h: int,
                          stabilized_left: Optional[FencerData],
                          stabilized_right: Optional[FencerData],
                          cam_tx: float,
                          cam_ty: float) -> np.ndarray:
        """Render one Matplotlib frame and return as BGR image array suitable for cv2.VideoWriter."""
        self._init_visualizer(frame_w, frame_h)

        # Update artists
        self._update_artist_for_fencer(stabilized_left, self.left_scatter, self.left_lines)
        self._update_artist_for_fencer(stabilized_right, self.right_scatter, self.right_lines)

        self.info_text.set_text(
            f"Frame: {frame_idx}   "
            f"Cum. Camera Offset: ({cam_tx:.1f}, {cam_ty:.1f})"
        )

        # Draw
        self.canvas.draw()
        buf = np.frombuffer(self.canvas.buffer_rgba(), dtype=np.uint8)
        w, h = self.canvas.get_width_height()
        img_rgba = buf.reshape((h, w, 4))
        img_rgb = img_rgba[:, :, :3]
        img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
        return img_bgr

    def analyze_video(self, video_path: str, output_path: str,
                      verify_video_path: Optional[str] = None):
        """Analyze video with improved stabilization and optionally render a verification video."""
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Could not open video: {video_path}")

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        logger.info(f"Processing {total_frames} frames from {video_path} at {fps:.2f} FPS, size {width}x{height}")

        # Optional verification writer
        verify_writer = None
        if verify_video_path:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            verify_writer = cv2.VideoWriter(verify_video_path, fourcc, fps, (width, height))
            if not verify_writer.isOpened():
                verify_writer = None
                logger.warning(f"Could not open verification writer at {verify_video_path}")

        # Data storage
        data = {
            'left_x': [], 'left_y': [],
            'right_x': [], 'right_y': [],
            'debug_camera_x': [], 'debug_camera_y': []
        }

        frame_count = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            # Detect fencers
            raw_fencers = self._run_yolo_inference(frame)

            # Get stabilization transform
            transform = self.stabilizer.process_frame(
                frame, [f.bbox for f in raw_fencers]
            )

            # Apply stabilization to keypoints
            stabilized_fencers = self._transform_keypoints(raw_fencers, transform)

            # Sort fencers left to right
            stabilized_fencers.sort(key=lambda f: (f.bbox[0] + f.bbox[2]) / 2)

            # Select left/right (or None)
            left_fencer = stabilized_fencers[0] if len(stabilized_fencers) >= 1 else None
            right_fencer = stabilized_fencers[1] if len(stabilized_fencers) >= 2 else None

            # Store data
            if left_fencer is not None:
                data['left_x'].append(left_fencer.keypoints[:, 0].tolist())
                data['left_y'].append(left_fencer.keypoints[:, 1].tolist())
            else:
                data['left_x'].append([np.nan] * 17)
                data['left_y'].append([np.nan] * 17)

            if right_fencer is not None:
                data['right_x'].append(right_fencer.keypoints[:, 0].tolist())
                data['right_y'].append(right_fencer.keypoints[:, 1].tolist())
            else:
                data['right_x'].append([np.nan] * 17)
                data['right_y'].append([np.nan] * 17)

            # Store camera motion (cumulative)
            cam_tx = float(transform[0, 2])
            cam_ty = float(transform[1, 2])
            data['debug_camera_x'].append(cam_tx)
            data['debug_camera_y'].append(cam_ty)

            # Render verification frame (stabilized keypoints in frame coordinates)
            if verify_writer is not None:
                viz_frame = self._render_viz_frame(
                    frame_idx=frame_count,
                    frame_w=width,
                    frame_h=height,
                    stabilized_left=left_fencer,
                    stabilized_right=right_fencer,
                    cam_tx=cam_tx,
                    cam_ty=cam_ty
                )
                # Ensure exact size for writer
                if viz_frame.shape[1] != width or viz_frame.shape[0] != height:
                    viz_frame = cv2.resize(viz_frame, (width, height), interpolation=cv2.INTER_LINEAR)
                verify_writer.write(viz_frame)

            frame_count += 1
            if total_frames > 0 and frame_count % 50 == 0:
                logger.info(f"Progress: {frame_count}/{total_frames} ({frame_count/total_frames*100:.1f}%)")

        cap.release()
        if verify_writer is not None:
            verify_writer.release()

        # Save to Excel
        with pd.ExcelWriter(output_path) as writer:
            pd.DataFrame(data['left_x'], columns=POSE_KEYPOINT_NAMES).to_excel(
                writer, sheet_name='Left_Fencer_X', index_label='Frame'
            )
            pd.DataFrame(data['left_y'], columns=POSE_KEYPOINT_NAMES).to_excel(
                writer, sheet_name='Left_Fencer_Y', index_label='Frame'
            )
            pd.DataFrame(data['right_x'], columns=POSE_KEYPOINT_NAMES).to_excel(
                writer, sheet_name='Right_Fencer_X', index_label='Frame'
            )
            pd.DataFrame(data['right_y'], columns=POSE_KEYPOINT_NAMES).to_excel(
                writer, sheet_name='Right_Fencer_Y', index_label='Frame'
            )
            pd.DataFrame({
                'Camera_Motion_X': data['debug_camera_x'],
                'Camera_Motion_Y': data['debug_camera_y']
            }).to_excel(writer, sheet_name='Debug_Camera_Motion', index_label='Frame')

        # Print summary
        debug_summary = self.stabilizer.get_debug_summary()
        print("\n" + "="*60)
        print("STABILIZATION COMPLETE")
        print("="*60)
        print(f"Frames processed: {debug_summary['frames_processed']}")
        print(f"Successful motion tracks: {debug_summary['successful_tracks']}")
        print(f"Strip line detections: {debug_summary['strip_detections']}")
        print(f"Success rate: {debug_summary['success_rate']:.1f}%")
        print(f"Total camera motion compensated:")
        print(f"  X-axis: {abs(debug_summary['total_motion_x']):.1f} pixels")
        print(f"  Y-axis: {abs(debug_summary['total_motion_y']):.1f} pixels")
        print(f"\nResults saved to: {output_path}")
        if verify_video_path:
            print(f"Verification preview saved to: {verify_video_path}")
        print("="*60)


if __name__ == "__main__":
    VIDEO_PATH = "./626_right.mp4"
    OUTPUT_PATH = "fencing_analysis_improved.xlsx"
    VERIFY_VIDEO_PATH = "stabilization_preview.mp4"   # Set to None to disable

    analyzer = FencingVideoAnalyzer()
    analyzer.analyze_video(VIDEO_PATH, OUTPUT_PATH, verify_video_path=VERIFY_VIDEO_PATH)
