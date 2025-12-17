# Match Separation in Jupyter Notebook
# This script processes keypoint data from a fencer tracking pipeline to detect fencing match start and end frames,
# saves match-specific data, and overlays normalized keypoints on match videos.

# Import necessary libraries
import numpy as np
import pandas as pd
import os
import cv2
import matplotlib.pyplot as plt
import torch
import torchvision
import math
import logging
import traceback
#%matplotlib inline

# Placeholder for custom module (unavailable); comment out or implement as needed
# from models import Bout

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s',
    handlers=[
        logging.FileHandler('match_separation.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger()

# User inputs (modify these in the notebook)
video_path = "./training videos/4980.MOV"  # Path to input video (same as fencer tracking script)
csv_output_dir = "output"  # Directory containing CSV files from fencer tracking script
matches_output_dir = "matches"  # Directory to save match videos
match_data_dir = "match_data"  # Directory to save match-specific CSV files
keypoints_output_dir = "keypoints_videos"  # Directory to save videos with keypoint overlays
fps = 30  # Output FPS (match with fencer tracking script)

# Utility functions
def load_data_from_csv(input_dir):
    try:
        logger.debug(f"Loading CSVs from {input_dir}")
        left_x_df = pd.read_csv(os.path.join(input_dir, 'left_xdata.csv'), dtype={str(i): float for i in range(7, 17)})
        left_y_df = pd.read_csv(os.path.join(input_dir, 'left_ydata.csv'), dtype={str(i): float for i in range(7, 17)})
        right_x_df = pd.read_csv(os.path.join(input_dir, 'right_xdata.csv'), dtype={str(i): float for i in range(7, 17)})
        right_y_df = pd.read_csv(os.path.join(input_dir, 'right_ydata.csv'), dtype={str(i): float for i in range(7, 17)})
        meta_df = pd.read_csv(os.path.join(input_dir, 'meta.csv'))
        required_columns = ['7', '8', '9', '10', '11', '12', '13', '14', '15', '16']
        for df, name in [(left_x_df, 'left_xdata'), (left_y_df, 'left_ydata'), (right_x_df, 'right_xdata'), (right_y_df, 'right_ydata')]:
            if not all(col in df.columns for col in required_columns):
                logger.error(f"Missing required keypoint columns in {name}: {df.columns}")
                raise ValueError(f"Missing required keypoint columns in {name}: {df.columns}")
            for col in required_columns:
                if not pd.api.types.is_numeric_dtype(df[col]):
                    logger.error(f"Non-numeric data in column {col} of {name}")
                    raise ValueError(f"Non-numeric data in column {col} of {name}")
        c = meta_df['c'][0]
        checker_list = eval(meta_df['checker_list'][0]) if meta_df['checker_list'][0] else []
        video_angle = meta_df['video_angle'][0]
        # Reset index to ensure 0-based indexing
        left_x_df = left_x_df.reset_index(drop=True)
        left_y_df = left_y_df.reset_index(drop=True)
        right_x_df = right_x_df.reset_index(drop=True)
        right_y_df = right_y_df.reset_index(drop=True)
        left_x_df['Frame'] = range(len(left_x_df))
        left_y_df['Frame'] = range(len(left_y_df))
        right_x_df['Frame'] = range(len(right_x_df))
        right_y_df['Frame'] = range(len(right_y_df))
        lengths = [len(left_x_df), len(left_y_df), len(right_x_df), len(right_y_df)]
        if len(set(lengths)) > 1:
            logger.error(f"Inconsistent dataset lengths: {lengths}")
            raise ValueError(f"Inconsistent dataset lengths: {lengths}")
        logger.info(f"Loaded datasets with scaling factor c={c}")
        logger.info(f"Dataset contains {len(left_x_df)} frames")
        logger.debug(f"left_x_df columns: {list(left_x_df.columns)}")
        logger.debug(f"left_x_df first row: {left_x_df.iloc[0].to_dict()}")
        return left_x_df, left_y_df, right_x_df, right_y_df, c, checker_list, video_angle
    except Exception as e:
        logger.error(f"Error loading CSV files: {str(e)}\n{traceback.format_exc()}")
        raise

def load_video_frames(video_path):
    try:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise FileNotFoundError(f"Could not open video file {video_path}")
        frames = []
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        cap.release()
        if not frames:
            raise ValueError(f"No frames read from video {video_path}")
        logger.info(f"Loaded {len(frames)} frames from {video_path}")
        return frames
    except Exception as e:
        logger.error(f"Error loading video: {e}\n{traceback.format_exc()}")
        raise

def generate_video_from_frames(frames, output_path, fps=30):
    logger.debug(f"Generating video at {output_path} with {len(frames)} frames")
    frames = torch.from_numpy(np.asarray(frames))
    if not os.path.exists(os.path.dirname(output_path)):
        os.makedirs(os.path.dirname(output_path))
    # Use H.264 codec and yuv420p pixel format for maximum web compatibility
    options = {'crf': '23', 'pix_fmt': 'yuv420p'}
    torchvision.io.write_video(output_path, frames, fps=fps, video_codec="libx264", options=options)
    logger.info(f"Video saved to {output_path}")
    return output_path

def extract_keypoints(xdata, ydata, index):
    try:
        logger.debug(f"Extracting keypoints at index {index}")
        if index not in xdata.index or index not in ydata.index:
            logger.error(f"Index {index} not found in DataFrame")
            raise ValueError(f"Index {index} not found in DataFrame")
        required_columns = ['10', '11', '12', '13', '14', '15', '16']
        for df, name in [(xdata, 'xdata'), (ydata, 'ydata')]:
            missing_cols = [col for col in required_columns if col not in df.columns]
            if missing_cols:
                logger.error(f"Missing columns in {name}: {missing_cols}")
                raise ValueError(f"Missing columns in {name}: {missing_cols}")
        keypoints = {
            'hipback': (xdata.loc[index, '11'], ydata.loc[index, '11']),
            'kneeback': (xdata.loc[index, '13'], ydata.loc[index, '13']),
            'ankleback': (xdata.loc[index, '15'], ydata.loc[index, '15']),
            'hipfront': (xdata.loc[index, '12'], ydata.loc[index, '12']),
            'kneefront': (xdata.loc[index, '14'], ydata.loc[index, '14']),
            'anklefront': (xdata.loc[index, '16'], ydata.loc[index, '16']),
            'handfront': (xdata.loc[index, '10'], ydata.loc[index, '10'])
        }
        for kp, (x, y) in keypoints.items():
            if x is None or y is None:
                logger.error(f"None value detected in keypoint {kp} at index {index}")
                raise ValueError(f"None value detected in keypoint {kp} at index {index}")
            if np.isnan(x) or np.isnan(y):
                logger.error(f"NaN detected in keypoint {kp} at index {index}: x={x}, y={y}")
                raise ValueError(f"NaN detected in keypoint {kp} at index {index}")
            if not (isinstance(x, (int, float)) and isinstance(y, (int, float))):
                logger.error(f"Non-numeric value in keypoint {kp} at index {index}: x={x}, y={y}")
                raise ValueError(f"Non-numeric value in keypoint {kp} at index {index}")
        logger.debug(f"Keypoints extracted: {keypoints}")
        return keypoints
    except Exception as e:
        logger.error(f"Error extracting keypoints at index {index}: {str(e)}\n{traceback.format_exc()}")
        raise ValueError(f"Error extracting keypoints at index {index}: {e}")

def calculate_angle(p1, p2, p3):
    try:
        a = np.array(p1)
        b = np.array(p2)
        c = np.array(p3)
        ab = a - b
        cb = c - b
        norm_ab = np.linalg.norm(ab)
        norm_cb = np.linalg.norm(cb)
        if norm_ab == 0 or norm_cb == 0:
            logger.warning("Zero norm detected in angle calculation")
            return 0.0
        cosine_angle = np.dot(ab, cb) / (norm_ab * norm_cb)
        angle = np.degrees(np.arccos(np.clip(cosine_angle, -1.0, 1.0)))
        return angle
    except Exception as e:
        logger.error(f"Error calculating angle: {e}\n{traceback.format_exc()}")
        return 0.0

def calculate_fencer_angles(fencer_keypoints):
    angles = {
        'back_knee_angle': calculate_angle(fencer_keypoints['ankleback'], fencer_keypoints['kneeback'], fencer_keypoints['hipback']),
        'back_hip_angle': calculate_angle(fencer_keypoints['kneeback'], fencer_keypoints['hipback'], fencer_keypoints['hipfront']),
        'front_hip_angle': calculate_angle(fencer_keypoints['hipback'], fencer_keypoints['hipfront'], fencer_keypoints['kneefront']),
        'front_knee_angle': calculate_angle(fencer_keypoints['hipfront'], fencer_keypoints['kneefront'], fencer_keypoints['anklefront']),
        'hand_hip_angle': calculate_angle(fencer_keypoints['handfront'], fencer_keypoints['hipfront'], fencer_keypoints['kneefront']),
    }
    logger.debug(f"Calculated angles: {angles}")
    return angles

def calculate_jerk(acceleration):
    jerk = np.diff(acceleration)
    logger.debug(f"Calculated jerk shape: {jerk.shape}")
    return jerk

def calculate_jerk_based_frames(left_xdata_df, right_xdata_df):
    """Calculate jerk and acceleration minima frames for hit detection."""

    if len(left_xdata_df) < 4 or len(right_xdata_df) < 4:
        logger.debug("Insufficient frames for jerk analysis; defaulting to empty set")
        return np.array([])

    left_velocity = np.diff(left_xdata_df['16'].values)
    right_velocity = -np.diff(right_xdata_df['16'].values)
    left_acceleration = np.diff(left_velocity)
    right_acceleration = np.diff(right_velocity)
    left_jerk = np.diff(left_acceleration)
    right_jerk = np.diff(right_acceleration)

    min_left_jerk_indices = np.argsort(left_jerk)[:3]
    min_right_jerk_indices = np.argsort(right_jerk)[:3]
    min_left_acceleration_indices = np.argsort(left_acceleration)[:3]
    min_right_acceleration_indices = np.argsort(right_acceleration)[:3]

    all_jerk_frames = np.concatenate([
        left_xdata_df.iloc[min_left_jerk_indices]['Frame'].values,
        right_xdata_df.iloc[min_right_jerk_indices]['Frame'].values,
        left_xdata_df.iloc[min_left_acceleration_indices]['Frame'].values,
        right_xdata_df.iloc[min_right_acceleration_indices]['Frame'].values
    ])

    logger.debug(f"Jerk-based candidate frames: {all_jerk_frames}")
    return all_jerk_frames

def merge_consecutive_numbers(numbers):
    if not numbers:
        return []
    numbers = sorted(numbers)
    merged_segments = []
    start = numbers[0]
    end = numbers[0]
    for i in range(1, len(numbers)):
        if numbers[i] == end + 1:
            end = numbers[i]
        else:
            merged_segments.append((start, end))
            start = numbers[i]
            end = numbers[i]
    merged_segments.append((start, end))
    logger.debug(f"Merged segments: {merged_segments}")
    return merged_segments

def filter_segments_by_jerk_frames(segments, jerk_frames):
    filtered_segments = []
    for seg in segments:
        if any(frame in jerk_frames for frame in range(seg[0] - 5, seg[1] + 5)):
            filtered_segments.append(seg)
    logger.debug(f"Filtered segments by jerk: {filtered_segments}")
    return filtered_segments

def find_longest_segment(segments):
    if not segments:
        return None
    segment_lengths = [(seg, seg[1] - seg[0] + 1) for seg in segments]
    longest_segment = max(segment_lengths, key=lambda x: x[1])[0]
    logger.debug(f"Longest segment: {longest_segment}")
    return longest_segment[0] + math.ceil((longest_segment[1] - longest_segment[0]) / 2)

def find_end_frame(left_x_slice, right_x_slice, jerk_frames, global_distance_threshold):
    """Locate bout end frame using combined distance and jerk cues."""

    if len(left_x_slice) < 10 or len(right_x_slice) < 10:
        logger.debug("Insufficient slice length (%d) for end-frame detection", len(left_x_slice))
        return None

    distance = np.abs(right_x_slice['16'] - left_x_slice['16']).values

    left_vel = np.abs(np.diff(left_x_slice['16'].values))
    right_vel = np.abs(np.diff(right_x_slice['16'].values))
    left_vel = np.insert(left_vel, 0, 0)
    right_vel = np.insert(right_vel, 0, 0)

    left_accel = np.diff(left_vel)
    right_accel = np.diff(right_vel)
    left_accel = np.insert(left_accel, 0, 0)
    right_accel = np.insert(right_accel, 0, 0)

    left_jerk = np.abs(np.diff(left_accel))
    right_jerk = np.abs(np.diff(right_accel))
    left_jerk = np.insert(left_jerk, 0, 0)
    right_jerk = np.insert(right_jerk, 0, 0)

    combined_jerk = left_jerk + right_jerk

    window = 3
    distance_smooth = np.convolve(distance, np.ones(window) / window, mode='same')
    jerk_smooth = np.convolve(combined_jerk, np.ones(window) / window, mode='same')

    close_frames = np.where(
        (distance_smooth < global_distance_threshold) | (distance_smooth < 0.7)
    )[0]

    if len(close_frames) == 0:
        logger.debug(
            "No close approach detected (threshold=%.3f, min distance=%.3f)",
            global_distance_threshold,
            float(distance_smooth.min()) if len(distance_smooth) else 0.0,
        )
        return None

    min_distance_frame = int(distance_smooth.argmin())
    logger.debug(
        "Minimum distance %.3f at frame %d within slice",
        distance_smooth[min_distance_frame],
        min_distance_frame,
    )

    jerk_threshold = np.percentile(jerk_smooth, 80) if len(jerk_smooth) else 0.0
    high_jerk_frames = (
        np.where(jerk_smooth > jerk_threshold)[0]
        if jerk_threshold > 0
        else np.array([], dtype=int)
    )

    if len(high_jerk_frames) == 0:
        logger.debug("No high-jerk frames above threshold %.4f", jerk_threshold)

    hit_candidates = []
    distance_denominator = max(global_distance_threshold, 1e-6)
    jerk_denominator = max(jerk_threshold, 1e-6)

    for close_frame in close_frames:
        nearby_jerk = [j for j in high_jerk_frames if abs(int(j) - int(close_frame)) <= 10]
        if nearby_jerk:
            score = (1.0 - distance_smooth[close_frame] / distance_denominator) + (
                jerk_smooth[close_frame] / jerk_denominator if jerk_threshold > 0 else 0.0
            )
            hit_candidates.append((int(close_frame), float(score), 'distance+jerk'))

    search_window = 15
    min_dist_start = max(0, min_distance_frame - search_window)
    min_dist_end = min(len(jerk_smooth), min_distance_frame + search_window)

    for frame in range(min_dist_start, min_dist_end):
        if jerk_smooth[frame] > jerk_threshold * 0.7:
            score = (1.0 - distance_smooth[frame] / distance_denominator) + (
                jerk_smooth[frame] / jerk_denominator if jerk_threshold > 0 else 0.0
            )
            hit_candidates.append((int(frame), float(score), 'min_distance_region'))

    combined_score = np.zeros(len(distance_smooth))
    mean_jerk = np.mean(jerk_smooth) if len(jerk_smooth) else 0.0
    for idx in range(len(combined_score)):
        dist_score = max(0.0, 1.0 - distance_smooth[idx] / distance_denominator)
        jerk_score = jerk_smooth[idx] / (mean_jerk + 1e-6)
        combined_score[idx] = dist_score * 0.6 + jerk_score * 0.4

    search_end = int(len(combined_score) * 0.67)
    if search_end > 20:
        peak_score_frame = int(combined_score[:search_end].argmax())
        if combined_score[peak_score_frame] > 0.5:
            hit_candidates.append(
                (peak_score_frame, float(combined_score[peak_score_frame]), 'combined_score')
            )

    if not hit_candidates:
        fallback = min(min_distance_frame + 15, len(left_x_slice) - 1)
        logger.debug("No hit candidates; falling back to frame %d", fallback)
        return fallback

    unique_candidates = {}
    for frame, score, method in hit_candidates:
        if frame not in unique_candidates or score > unique_candidates[frame][0]:
            unique_candidates[frame] = (score, method)

    best_frame, (hit_score, hit_method) = max(
        unique_candidates.items(), key=lambda item: item[1][0]
    )

    buffer_frames = 10
    end_frame = min(best_frame + buffer_frames, len(left_x_slice) - 1)

    logger.info(
        "Hit detected at frame %d via %s (score=%.2f); end frame %d",
        best_frame,
        hit_method,
        hit_score,
        end_frame,
    )
    return end_frame

def save_match_data(left_xdata_df, left_ydata_df, right_xdata_df, right_ydata_df, start_frame, end_frame, match_idx, output_dir):
    match_dir = os.path.join(output_dir, f"match_{match_idx}")
    if not os.path.exists(match_dir):
        os.makedirs(match_dir)
    slice_indices = left_xdata_df[(left_xdata_df['Frame'] >= start_frame) & (left_xdata_df['Frame'] <= end_frame)].index
    if len(slice_indices) == 0:
        logger.warning(f"No data for match {match_idx} between frames {start_frame} and {end_frame}")
        return
    left_x_slice = left_xdata_df.loc[slice_indices].copy()
    left_y_slice = left_ydata_df.loc[slice_indices].copy()
    right_x_slice = right_xdata_df.loc[slice_indices].copy()
    right_y_slice = right_ydata_df.loc[slice_indices].copy()
    left_x_slice['Frame'] = left_x_slice['Frame'] - start_frame
    left_y_slice['Frame'] = left_y_slice['Frame'] - start_frame
    right_x_slice['Frame'] = right_x_slice['Frame'] - start_frame
    right_y_slice['Frame'] = right_y_slice['Frame'] - start_frame
    left_x_slice.to_csv(os.path.join(match_dir, 'left_xdata.csv'), index=False)
    left_y_slice.to_csv(os.path.join(match_dir, 'left_ydata.csv'), index=False)
    right_x_slice.to_csv(os.path.join(match_dir, 'right_xdata.csv'), index=False)
    right_y_slice.to_csv(os.path.join(match_dir, 'right_ydata.csv'), index=False)
    logger.info(f"Saved match {match_idx} data to {match_dir}")

def load_match_data(match_idx, match_data_dir):
    match_dir = os.path.join(match_data_dir, f"match_{match_idx}")
    try:
        left_x_df = pd.read_csv(os.path.join(match_dir, 'left_xdata.csv'))
        left_y_df = pd.read_csv(os.path.join(match_dir, 'left_ydata.csv'))
        right_x_df = pd.read_csv(os.path.join(match_dir, 'right_xdata.csv'))
        right_y_df = pd.read_csv(os.path.join(match_dir, 'right_ydata.csv'))
        required_columns = ['7', '8', '9', '10', '11', '12', '13', '14', '15', '16', 'Frame']
        for df in [left_x_df, left_y_df, right_x_df, right_y_df]:
            if not all(col in df.columns for col in required_columns):
                raise ValueError(f"Missing required columns in CSV: {df.columns}")
            for col in required_columns[:-1]:
                if not pd.api.types.is_numeric_dtype(df[col]):
                    raise ValueError(f"Non-numeric data in column {col} of DataFrame")
        logger.info(f"Loaded match {match_idx} data from {match_dir}")
        return left_x_df, left_y_df, right_x_df, right_y_df
    except Exception as e:
        logger.error(f"Error loading match {match_idx} data: {e}\n{traceback.format_exc()}")
        return None, None, None, None

def overlay_keypoints_on_clip(match_idx, start_frame, end_frame, left_x_df, left_y_df, right_x_df, right_y_df, c, input_clip_path, output_dir, fps=30):
    logger.debug(f"Overlaying keypoints for match {match_idx} from frame {start_frame} to {end_frame}")
    cap = cv2.VideoCapture(input_clip_path)
    if not cap.isOpened():
        logger.error(f"Could not open input clip {input_clip_path}")
        return
    frames = []
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    frame_idx = 0
    # Extend video by 1 second (fps frames), but cap at video length or next match start
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    next_start = start_frame + len(left_x_df) if len(left_x_df) > 0 else end_frame + 1
    video_end_frame = min(end_frame + fps, next_start - 1, total_frames - 1)
    while frame_idx <= video_end_frame - start_frame:
        ret, frame = cap.read()
        if not ret:
            logger.warning(f"Reached end of video at frame {start_frame + frame_idx}")
            break
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        # Only apply keypoints up to original end_frame (data availability)
        if frame_idx < len(left_x_df):
            for kp in range(7, 17):
                try:
                    x_left = left_x_df.loc[frame_idx, str(kp)] if not np.isnan(left_x_df.loc[frame_idx, str(kp)]) else None
                    y_left = left_y_df.loc[frame_idx, str(kp)] if not np.isnan(left_y_df.loc[frame_idx, str(kp)]) else None
                    if x_left is not None and y_left is not None:
                        kp_x = int(x_left * c)
                        kp_y = int(y_left * c)
                        cv2.circle(frame_rgb, (kp_x, kp_y), 5, (0, 0, 255), -1)  # Red for left fencer
                        cv2.putText(frame_rgb, f"{kp}", (kp_x + 10, kp_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
                    x_right = right_x_df.loc[frame_idx, str(kp)] if not np.isnan(right_x_df.loc[frame_idx, str(kp)]) else None
                    y_right = right_y_df.loc[frame_idx, str(kp)] if not np.isnan(right_y_df.loc[frame_idx, str(kp)]) else None
                    if x_right is not None and y_right is not None:
                        kp_x = int(x_right * c)
                        kp_y = int(y_right * c)
                        cv2.circle(frame_rgb, (kp_x, kp_y), 5, (255, 0, 0), -1)  # Blue for right fencer
                        cv2.putText(frame_rgb, f"{kp}", (kp_x + 10, kp_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
                except KeyError:
                    logger.warning(f"Keypoint {kp} or frame {frame_idx} not found in DataFrame")
                    continue
        frames.append(frame_rgb)
        frame_idx += 1
    cap.release()
    if frames:
        output_path = os.path.join(output_dir, f"match_{match_idx}_keypoints.mp4")
        generate_video_from_frames(frames, output_path, fps=fps)
        logger.info(f"Overlayed keypoints video for match {match_idx} saved to {output_path} (extended to frame {video_end_frame})")
    else:
        logger.warning(f"No frames processed for match {match_idx} overlay video")
        
def detect_start_frames(
    left_xdata_df,
    left_ydata_df,
    right_xdata_df,
    right_ydata_df,
    static_velocity_threshold=0.01,
    movement_velocity_threshold=0.03,
    min_static_frames=15,
    min_movement_frames=3,
    min_frame_spacing=100,
):
    """Detect match start frames using velocity spikes after static periods."""

    left_foot = left_xdata_df['16'].values if '16' in left_xdata_df.columns else left_xdata_df.iloc[:, 16].values
    right_foot = right_xdata_df['16'].values if '16' in right_xdata_df.columns else right_xdata_df.iloc[:, 16].values

    left_vel = np.abs(np.diff(left_foot))
    right_vel = np.abs(np.diff(right_foot))
    left_vel = np.insert(left_vel, 0, 0)
    right_vel = np.insert(right_vel, 0, 0)

    window = 3
    left_vel_smooth = np.convolve(left_vel, np.ones(window) / window, mode='same')
    right_vel_smooth = np.convolve(right_vel, np.ones(window) / window, mode='same')
    combined_vel = np.maximum(left_vel_smooth, right_vel_smooth)

    candidate_frames = []
    i = 0
    num_frames = len(combined_vel)

    while i < num_frames:
        if combined_vel[i] <= static_velocity_threshold:
            static_count = 0
            j = i
            static_avg_vel = []

            while j < num_frames and combined_vel[j] <= static_velocity_threshold:
                static_avg_vel.append(combined_vel[j])
                static_count += 1
                j += 1

            if static_count >= min_static_frames:
                baseline_vel = float(np.mean(static_avg_vel)) if static_avg_vel else 0.0
                logger.debug(
                    "Static segment %d-%d (%d frames, baseline vel=%.4f)",
                    i,
                    j - 1,
                    static_count,
                    baseline_vel,
                )

                movement_found = False
                lookahead = 20

                for look_frame in range(j, min(j + lookahead, num_frames)):
                    if combined_vel[look_frame] >= movement_velocity_threshold:
                        peak_frame = look_frame
                        peak_vel = combined_vel[look_frame]

                        for check_frame in range(look_frame, min(look_frame + 5, num_frames)):
                            if combined_vel[check_frame] > peak_vel:
                                peak_vel = combined_vel[check_frame]
                                peak_frame = check_frame

                        sustained_movement = sum(
                            1
                            for check_frame in range(peak_frame, min(peak_frame + min_movement_frames, num_frames))
                            if combined_vel[check_frame] >= movement_velocity_threshold * 0.7
                        )

                        if sustained_movement >= min_movement_frames:
                            takeoff_frame = peak_frame
                            for back_frame in range(peak_frame, max(j - 1, look_frame - 3), -1):
                                if combined_vel[back_frame] < movement_velocity_threshold * 0.5:
                                    takeoff_frame = back_frame + 1
                                    break
                                takeoff_frame = back_frame

                            candidate_frames.append(takeoff_frame)
                            logger.info(
                                "Static segment (%d,%d): takeoff frame %d (peak=%d, vel=%.4f)",
                                i,
                                j - 1,
                                takeoff_frame,
                                peak_frame,
                                peak_vel,
                            )
                            movement_found = True
                            break

                if not movement_found:
                    logger.debug("Static segment (%d,%d): no qualifying movement spike", i, j - 1)

                i = j
            else:
                i += 1
        else:
            i += 1

    if not candidate_frames:
        logger.warning("No candidate start frames detected; defaulting to frame 0")
        return [0]

    candidate_frames = sorted(set(candidate_frames))
    start_frames = [candidate_frames[0]]
    for frame in candidate_frames[1:]:
        if frame - start_frames[-1] >= min_frame_spacing:
            start_frames.append(frame)

    logger.info(f"Filtered start frames: {start_frames}")
    return start_frames

def plot_end_frames(video_path, left_xdata_df, left_ydata_df, right_xdata_df, right_ydata_df, c, start_frames, end_frames):
    frames = load_video_frames(video_path)
    logger.info(f"Loaded video with {len(frames)} frames")
    for start_frame, end_frame in zip(start_frames, end_frames):
        if end_frame >= len(frames):
            logger.warning(f"End frame {end_frame} exceeds video length")
            continue
        frame = frames[end_frame]
        fig, ax = plt.subplots(figsize=(12, 8))
        ax.imshow(frame)
        if end_frame < len(left_xdata_df):
            for kp in range(7, 17):
                try:
                    x = left_xdata_df.loc[end_frame, str(kp)] if not np.isnan(left_xdata_df.loc[end_frame, str(kp)]) else None
                    y = left_ydata_df.loc[end_frame, str(kp)] if not np.isnan(left_ydata_df.loc[end_frame, str(kp)]) else None
                    if x is not None and y is not None:
                        kp_x = x * c
                        kp_y = y * c
                        ax.scatter(kp_x, kp_y, c='red', s=50, label='Left Fencer' if kp == 7 else "")
                        ax.text(kp_x + 10, kp_y, f"{kp}:({x:.2f},{y:.2f})", color='red', fontsize=8)
                    x = right_xdata_df.loc[end_frame, str(kp)] if not np.isnan(right_xdata_df.loc[end_frame, str(kp)]) else None
                    y = right_ydata_df.loc[end_frame, str(kp)] if not np.isnan(right_ydata_df.loc[end_frame, str(kp)]) else None
                    if x is not None and y is not None:
                        kp_x = x * c
                        kp_y = y * c
                        ax.scatter(kp_x, kp_y, c='blue', s=50, label='Right Fencer' if kp == 7 else "")
                        ax.text(kp_x + 10, kp_y, f"{kp}:({x:.2f},{y:.2f})", color='blue', fontsize=8)
                except KeyError:
                    logger.warning(f"Keypoint {kp} not found at frame {end_frame}")
                    continue
        ax.set_title(f"End Frame {end_frame} for Start Frame {start_frame}")
        ax.axis('off')
        ax.legend(loc='upper right')
        output_path = os.path.join(os.path.dirname(video_path), f"end_frame_{end_frame}.png")
        plt.savefig(output_path)
        plt.show()
        plt.close()
        logger.info(f"Generated plot for end frame {end_frame} at {output_path}")

def detect_matches(
    left_xdata_df,
    left_ydata_df,
    right_xdata_df,
    right_ydata_df,
    max_frame,
    video_path,
    c,
    output_dir,
    match_data_dir,
    keypoints_output_dir,
    fps=30,
    min_match_duration=10,
):
    start_frames = detect_start_frames(left_xdata_df, left_ydata_df, right_xdata_df, right_ydata_df)
    if not start_frames:
        logger.warning("No start frames detected; aborting match detection")
        return []

    matches = []
    all_jerk_frames = calculate_jerk_based_frames(left_xdata_df, right_xdata_df)
    logger.debug(f"All jerk-based frames: {all_jerk_frames}")

    distance_series = np.abs(right_xdata_df['16'] - left_xdata_df['16']).values
    if len(distance_series) == 0:
        logger.error("Distance series empty; cannot derive end-frame thresholds")
        return matches

    global_distance_threshold = float(np.percentile(distance_series, 7))
    min_global_distance = float(np.min(distance_series))
    logger.info(
        "Global distance threshold %.3f (min distance %.3f)",
        global_distance_threshold,
        min_global_distance,
    )

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        logger.error(f"Could not open video file {video_path}")
        return matches

    total_video_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    max_video_index = total_video_frames - 1 if total_video_frames > 0 else max_frame

    for idx, start_frame in enumerate(start_frames):
        next_start = start_frames[idx + 1] if idx + 1 < len(start_frames) else max_frame + 1
        window_end = min(next_start, max_frame + 1)
        slice_indices = left_xdata_df[
            (left_xdata_df['Frame'] >= start_frame) & (left_xdata_df['Frame'] < window_end)
        ].index

        if len(slice_indices) < min_match_duration:
            logger.info(
                "Skipping start frame %d: only %d frames before next bout (min %d)",
                start_frame,
                len(slice_indices),
                min_match_duration,
            )
            continue

        left_x_slice = left_xdata_df.loc[slice_indices].reset_index(drop=True)
        right_x_slice = right_xdata_df.loc[slice_indices].reset_index(drop=True)
        left_y_slice = left_ydata_df.loc[slice_indices].reset_index(drop=True)
        right_y_slice = right_ydata_df.loc[slice_indices].reset_index(drop=True)

        left_x_slice['Frame'] = left_x_slice['Frame'] - start_frame
        right_x_slice['Frame'] = right_x_slice['Frame'] - start_frame

        slice_jerk_frames = [int(f - start_frame) for f in all_jerk_frames if start_frame <= f < window_end]

        end_frame_idx = find_end_frame(left_x_slice, right_x_slice, slice_jerk_frames, global_distance_threshold)
        if end_frame_idx is None:
            logger.info("Start frame %d: no valid end frame detected", start_frame)
            continue

        end_frame = start_frame + int(end_frame_idx)
        end_frame = min(end_frame, window_end - 1, max_frame)
        match_duration = end_frame - start_frame + 1

        if match_duration < min_match_duration:
            logger.debug(
                "Start frame %d: match duration %d below minimum %d",
                start_frame,
                match_duration,
                min_match_duration,
            )
            continue

        match_idx = len(matches) + 1
        matches.append((start_frame, end_frame))
        logger.info(
            "Match %d: frames %d-%d (%d frames, %.1fs)",
            match_idx,
            start_frame,
            end_frame,
            match_duration,
            match_duration / fps,
        )

        match_dir = os.path.join(output_dir, f"match_{match_idx}")
        os.makedirs(match_dir, exist_ok=True)

        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        video_end_frame = min(end_frame, window_end - 1, max_frame)

        frames = []
        for frame_idx in range(start_frame, video_end_frame + 1):
            ret, frame = cap.read()
            if not ret:
                logger.warning(
                    "Could not read frame %d for match %d", frame_idx, match_idx
                )
                break
            frames.append(frame)

        if frames:
            output_path = os.path.join(match_dir, f"match_{match_idx}.mp4")
            generate_video_from_frames(frames, output_path, fps=fps)
            logger.info(
                "Match %d video saved to %s (frames %d-%d)",
                match_idx,
                output_path,
                start_frame,
                start_frame + len(frames) - 1,
            )

        extended_output_path = os.path.join(match_dir, f"match_{match_idx}_extended.mp4")
        padding_frames = fps
        extended_start_frame = max(0, start_frame - padding_frames)
        extended_end_frame = min(video_end_frame + padding_frames, max_frame, max_video_index)

        cap.set(cv2.CAP_PROP_POS_FRAMES, extended_start_frame)
        extended_frames = []
        for frame_idx in range(extended_start_frame, extended_end_frame + 1):
            ret, frame = cap.read()
            if not ret:
                logger.warning(
                    "Could not read frame %d for extended match %d", frame_idx, match_idx
                )
                break
            extended_frames.append(frame)

        if extended_frames:
            generate_video_from_frames(extended_frames, extended_output_path, fps=fps)
            logger.info(
                "Match %d extended video saved to %s (frames %d-%d)",
                match_idx,
                extended_output_path,
                extended_start_frame,
                extended_start_frame + len(extended_frames) - 1,
            )

        save_match_data(
            left_xdata_df,
            left_ydata_df,
            right_xdata_df,
            right_ydata_df,
            start_frame,
            end_frame,
            match_idx,
            match_data_dir,
        )

        overlay_keypoints_on_clip(
            match_idx,
            start_frame,
            end_frame,
            left_x_slice,
            left_y_slice,
            right_x_slice,
            right_y_slice,
            c,
            video_path,
            keypoints_output_dir,
            fps,
        )

    cap.release()

    return matches

def main(video_path, csv_output_dir, matches_output_dir, match_data_dir, keypoints_output_dir, fps=30):
    logger.info(f"Starting match separation for video: {video_path}")
    try:
        left_xdata_df, left_ydata_df, right_xdata_df, right_ydata_df, c, checker_list, video_angle = load_data_from_csv(csv_output_dir)
        if left_xdata_df is None:
            logger.error("Failed to load CSV data")
            return [], 'unknown'
        max_frame = int(max(left_xdata_df['Frame'].max(), right_xdata_df['Frame'].max()))
        matches = detect_matches(
            left_xdata_df, left_ydata_df, right_xdata_df, right_ydata_df, 
            max_frame, video_path, c, matches_output_dir, match_data_dir, 
            keypoints_output_dir, fps
        )
        
        logger.info(f"Detected {len(matches)} matches: {matches}")
        return matches, video_angle
    except Exception as e:
        logger.error(f"Main processing failed: {str(e)}\n{traceback.format_exc()}")
        return [], 'unknown'
