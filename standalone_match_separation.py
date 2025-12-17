#!/usr/bin/env python3
"""
Standalone Match Separation Script

This script processes keypoint data from a fencer tracking pipeline to detect fencing match 
start and end frames, saves match-specific data, and overlays normalized keypoints on match videos.

Usage:
    python standalone_match_separation.py --video <video_path> --csv <csv_dir> --output <output_dir> [--fps <fps>]

Example:
    python standalone_match_separation.py --video video.mp4 --csv ./output --output ./results --fps 30
"""

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
import argparse

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
    left_velocity = np.diff(left_xdata_df['16'].values)
    right_velocity = -np.diff(right_xdata_df['16'].values)
    left_acceleration = np.diff(left_velocity)
    right_acceleration = np.diff(right_velocity)
    left_jerk = calculate_jerk(left_acceleration)
    right_jerk = calculate_jerk(right_acceleration)
    min_left_jerk_indices = np.argsort(left_jerk)[:3]
    min_right_jerk_indices = np.argsort(right_jerk)[:3]
    min_left_acceleration_indices = np.argsort(left_acceleration)[:3]
    min_right_acceleration_indices = np.argsort(right_acceleration)[:3]
    min_left_jerk_frames = left_xdata_df.iloc[min_left_jerk_indices]['Frame'].values
    min_right_jerk_frames = right_xdata_df.iloc[min_right_jerk_indices]['Frame'].values
    min_left_acceleration_frames = left_xdata_df.iloc[min_left_acceleration_indices]['Frame'].values
    min_right_acceleration_frames = right_xdata_df.iloc[min_right_acceleration_indices]['Frame'].values
    all_jerk_frames = np.concatenate((min_left_jerk_frames, min_right_jerk_frames, min_left_acceleration_frames, min_right_acceleration_frames))
    logger.debug(f"Jerk-based frames: {all_jerk_frames}")
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

def find_end_frame(left_xdata_new_df, right_xdata_new_df, all_jerk_frames):
    a = abs(right_xdata_new_df['16'] - left_xdata_new_df['16'])
    end_frames = []
    for i in range(len(a)):
        try:
            if 0 < a[i] < 0.5:
                end_frames.append(i)
        except:
            continue
    if not end_frames:
        thresh = a.quantile(0.10)
        logger.info(f"Using distance threshold {thresh:.3f}")
        for i in range(len(a)):
            try:
                if 0 < a[i] < thresh:
                    end_frames.append(i)
            except:
                continue
    end_segments = merge_consecutive_numbers(end_frames)
    logger.info(f"End segments: {end_segments}")
    filtered_end_segments = filter_segments_by_jerk_frames(end_segments, all_jerk_frames)
    logger.info(f"Filtered end segments: {filtered_end_segments}")
    if not filtered_end_segments:
        filtered_end_segments = end_segments[-1:] if end_segments else []
    if isinstance(filtered_end_segments, tuple):
        longest_segment = filtered_end_segments
        end_frame = longest_segment[0] + math.ceil((longest_segment[1] - longest_segment[0]) / 2)
    else:
        end_frame = find_longest_segment(filtered_end_segments)
    end_frame = end_frame if end_frame else left_xdata_new_df['Frame'].iloc[-1]
    logger.info(f"Selected end frame index: {end_frame}")
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
        
def detect_start_frames(left_xdata_df, left_ydata_df, right_xdata_df, right_ydata_df, movement_threshold=0.01, static_threshold=0.01, min_static_frames=5, min_moving_frames=5, static_window=5, lookahead_frames=10, min_frame_spacing=120, angle_diff_threshold=180, distance_tolerance=0.2):
    # Calculate velocities
    left_front_foot = left_xdata_df['16'].values
    right_front_foot = right_xdata_df['16'].values
    left_v = np.diff(left_front_foot)
    right_v = np.diff(right_front_foot)
    left_v = np.insert(left_v, 0, 0)
    right_v = np.insert(right_v, 0, 0)
    
    # Find first movement frame
    first_move_frame = None
    for frame in range(1, len(left_xdata_df)):
        try:
            if abs(left_v[frame]) > 0.1 or abs(right_v[frame]) > 0.1:
                first_move_frame = frame
                logger.info(f"First movement detected at frame {frame}: Left v={left_v[frame]:.3f}, Right v={right_v[frame]:.3f}")
                break
        except IndexError:
            continue
    
    # Backtrack 15 frames for reference
    if first_move_frame is not None:
        ref_frame = max(0, first_move_frame - 15)
        logger.info(f"Selected reference frame {ref_frame} (backtracked 15 frames from {first_move_frame})")
    else:
        ref_frame = 0
        logger.warning("No movement detected; defaulting to frame 0 as reference")
    
    # Set reference pose
    try:
        left_fencer_keypoints_ref = extract_keypoints(left_xdata_df, left_ydata_df, ref_frame)
        right_fencer_keypoints_ref = extract_keypoints(right_xdata_df, right_ydata_df, ref_frame)
        left_angles_ref = calculate_fencer_angles(left_fencer_keypoints_ref)
        right_angles_ref = calculate_fencer_angles(right_fencer_keypoints_ref)
        ref_distance = abs(left_xdata_df.loc[ref_frame, '16'] - right_xdata_df.loc[ref_frame, '16'])
        logger.info(f"Reference pose angles: Left={left_angles_ref}, Right={right_angles_ref}")
        logger.info(f"Reference distance between fencers: {ref_distance:.3f}")
    except Exception as e:
        logger.error(f"Error extracting reference pose from frame {ref_frame}: {e}\n{traceback.format_exc()}")
        return [0]
    
    # Continue with original start frame detection logic
    candidate_start_frames = [0]
    static_start = None
    static_count = 0
    for frame in range(1, len(left_xdata_df)):
        try:
            left_fencer_keypoints = extract_keypoints(left_xdata_df, left_ydata_df, frame)
            right_fencer_keypoints = extract_keypoints(right_xdata_df, right_ydata_df, frame)
            left_angles = calculate_fencer_angles(left_fencer_keypoints)
            right_angles = calculate_fencer_angles(right_fencer_keypoints)
            total_diff = 0
            for angle_name in left_angles_ref.keys():
                left_diff = abs(left_angles[angle_name] - left_angles_ref[angle_name])
                right_diff = abs(right_angles[angle_name] - right_angles_ref[angle_name])
                total_diff += left_diff + right_diff
            current_distance = abs(left_xdata_df.loc[frame, '16'] - right_xdata_df.loc[frame, '16'])
            distance_ratio = current_distance / ref_distance if ref_distance > 0 else 1.0
            distance_similar = 1.0 - distance_tolerance <= distance_ratio <= 1.0 + distance_tolerance
            window_start = max(0, frame - static_window + 1)
            window_left_v = left_v[window_start:frame + 1]
            window_right_v = right_v[window_start:frame + 1]
            avg_left_vel = np.mean(np.abs(window_left_v)) if len(window_left_v) > 0 else abs(left_v[frame])
            avg_right_vel = np.mean(np.abs(window_right_v)) if len(window_right_v) > 0 else abs(right_v[frame])
            if total_diff < angle_diff_threshold and distance_similar and avg_left_vel < static_threshold and avg_right_vel < static_threshold:
                if static_start is None:
                    static_start = frame
                static_count += 1
            else:
                if static_start is not None and static_count >= min_static_frames:
                    for next_frame in range(frame, min(frame + lookahead_frames, len(left_xdata_df))):
                        moving_count = 0
                        for check_frame in range(next_frame, min(next_frame + min_moving_frames, len(left_xdata_df))):
                            try:
                                check_left_vel = left_v[check_frame]
                                check_right_vel = right_v[check_frame]
                                if (check_left_vel > movement_threshold or check_right_vel < -movement_threshold):
                                    moving_count += 1
                            except IndexError:
                                break
                        if moving_count >= min_moving_frames:
                            candidate_start_frames.append(next_frame)
                            logger.info(f"Static segment ({static_start},{frame-1}): Candidate start frame {next_frame}")
                            break
                static_start = None
                static_count = 0
            logger.debug(f"Frame {frame}: Left v={left_v[frame]:.3f}, Right v={right_v[frame]:.3f}")
        except Exception as e:
            logger.warning(f"Frame {frame}: Skipped due to error: {e}")
            continue
    start_frames = []
    candidate_start_frames = sorted(list(set(candidate_start_frames)))
    logger.info(f"Candidate start frames before filtering: {candidate_start_frames}")
    i = 0
    while i < len(candidate_start_frames):
        current_frame = candidate_start_frames[i]
        keep_current = True
        for j in range(i + 1, len(candidate_start_frames)):
            if candidate_start_frames[j] - current_frame <= min_frame_spacing:
                keep_current = False
                break
        if keep_current:
            start_frames.append(current_frame + 10)
        i += 1
    logger.info(f"Final start frames: {start_frames}")
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

def detect_matches(left_xdata_df, left_ydata_df, right_xdata_df, right_ydata_df, max_frame, video_path, c, output_dir, match_data_dir, keypoints_output_dir, fps=30):
    start_frames = detect_start_frames(left_xdata_df, left_ydata_df, right_xdata_df, right_ydata_df)
    matches = []
    all_jerk_frames = calculate_jerk_based_frames(left_xdata_df, right_xdata_df)
    logger.debug(f"All jerk-based frames: {all_jerk_frames}")

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        logger.error(f"Could not open video file {video_path}")
        return matches
    
    for i, start_frame in enumerate(start_frames):
        next_start = start_frames[i + 1] if i + 1 < len(start_frames) else max_frame + 1
        slice_indices = left_xdata_df[(left_xdata_df['Frame'] >= start_frame) & (left_xdata_df['Frame'] < next_start)].index
        if len(slice_indices) == 0:
            logger.warning(f"No data between start frame {start_frame} and next start frame {next_start}")
            end_frame = max_frame
        else:
            left_x_slice = left_xdata_df.loc[slice_indices].reset_index(drop=True)
            right_x_slice = right_xdata_df.loc[slice_indices].reset_index(drop=True)
            left_x_slice['Frame'] = left_x_slice['Frame'] - start_frame
            right_x_slice['Frame'] = right_x_slice['Frame'] - start_frame
            slice_jerk_frames = [f - start_frame for f in all_jerk_frames if start_frame <= f < next_start]
            try:
                end_frame_idx = find_end_frame(left_x_slice, right_x_slice, slice_jerk_frames)
                end_frame = start_frame + end_frame_idx
                end_frame = min(end_frame, next_start - 1, max_frame)
            except Exception as e:
                logger.error(f"Error finding end frame for start frame {start_frame}: {str(e)}")
                end_frame = min(start_frame + 120, next_start - 1, max_frame)
        matches.append((start_frame, end_frame))
        logger.info(f"Match {i + 1}: Start frame {start_frame}, End frame {end_frame}")

        # Generate match video (original for data analysis)
        output_path = os.path.join(output_dir, f"match_{i + 1}", f"match_{i + 1}.mp4")
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # Get frames for the match
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        video_end_frame = min(end_frame, next_start - 1, max_frame)
        
        frames = []
        for frame_idx in range(start_frame, video_end_frame + 1):
            ret, frame = cap.read()
            if not ret:
                logger.warning(f"Could not read frame {frame_idx} for match {i + 1}")
                break
            frames.append(frame)
        
        # Generate video with the correct codec
        if frames:
            generate_video_from_frames(frames, output_path, fps=fps)
            logger.info(f"Match {i + 1} video saved to {output_path} (extended to frame {video_end_frame})")
        
        # Generate extended match video for display (with 1s padding on each side)
        extended_output_path = os.path.join(output_dir, f"match_{i + 1}", f"match_{i + 1}_extended.mp4")
        padding_frames = fps  # 1 second padding = fps frames
        extended_start_frame = max(0, start_frame - padding_frames)
        extended_end_frame = min(max_frame, video_end_frame + padding_frames)
        
        cap.set(cv2.CAP_PROP_POS_FRAMES, extended_start_frame)
        extended_frames = []
        for frame_idx in range(extended_start_frame, extended_end_frame + 1):
            ret, frame = cap.read()
            if not ret:
                logger.warning(f"Could not read frame {frame_idx} for extended match {i + 1}")
                break
            extended_frames.append(frame)
        
        # Generate extended video with the correct codec
        if extended_frames:
            generate_video_from_frames(extended_frames, extended_output_path, fps=fps)
            logger.info(f"Match {i + 1} extended video saved to {extended_output_path} (frames {extended_start_frame} to {extended_end_frame})")
        
        # Save match data
        save_match_data(left_xdata_df, left_ydata_df, right_xdata_df, right_ydata_df, start_frame, end_frame, i + 1, match_data_dir)
        
        # Overlay keypoints on match video
        left_x_slice = left_xdata_df.loc[slice_indices].reset_index(drop=True)
        left_y_slice = left_ydata_df.loc[slice_indices].reset_index(drop=True)
        right_x_slice = right_xdata_df.loc[slice_indices].reset_index(drop=True)
        right_y_slice = right_ydata_df.loc[slice_indices].reset_index(drop=True)
        overlay_keypoints_on_clip(i + 1, start_frame, end_frame, left_x_slice, left_y_slice, right_x_slice, right_y_slice, c, video_path, keypoints_output_dir, fps)
    
    cap.release()
    
    return matches

def main(video_path, csv_output_dir, output_dir, fps=30):
    """
    Main function to run match separation.
    
    Args:
        video_path: Path to the input video file
        csv_output_dir: Directory containing CSV files (left_xdata.csv, left_ydata.csv, etc.)
        output_dir: Base output directory for all results
        fps: Frames per second (default: 30)
    
    Returns:
        Tuple of (matches, video_angle) where matches is a list of (start_frame, end_frame) tuples
    """
    logger.info(f"Starting match separation for video: {video_path}")
    
    # Create output directories
    matches_output_dir = os.path.join(output_dir, "matches")
    match_data_dir = os.path.join(output_dir, "match_data")
    keypoints_output_dir = os.path.join(output_dir, "matches_with_keypoints")
    
    os.makedirs(matches_output_dir, exist_ok=True)
    os.makedirs(match_data_dir, exist_ok=True)
    os.makedirs(keypoints_output_dir, exist_ok=True)
    
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

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Standalone Match Separation Script for Fencing Videos',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example:
    python standalone_match_separation.py --video video.mp4 --csv ./output --output ./results --fps 30
    
The script expects the following CSV files in the csv directory:
    - left_xdata.csv
    - left_ydata.csv
    - right_xdata.csv
    - right_ydata.csv
    - meta.csv
    
Output will be created in the following structure:
    output_dir/
        matches/
            match_1/
                match_1.mp4
                match_1_extended.mp4
            match_2/
                ...
        match_data/
            match_1/
                left_xdata.csv
                left_ydata.csv
                right_xdata.csv
                right_ydata.csv
            match_2/
                ...
        matches_with_keypoints/
            match_1_keypoints.mp4
            match_2_keypoints.mp4
            ...
        """
    )
    
    parser.add_argument('--video', required=True, help='Path to input video file')
    parser.add_argument('--csv', required=True, help='Directory containing CSV files')
    parser.add_argument('--output', required=True, help='Output directory for results')
    parser.add_argument('--fps', type=int, default=30, help='Frames per second (default: 30)')
    
    args = parser.parse_args()
    
    # Validate inputs
    if not os.path.exists(args.video):
        logger.error(f"Video file not found: {args.video}")
        exit(1)
    
    if not os.path.exists(args.csv):
        logger.error(f"CSV directory not found: {args.csv}")
        exit(1)
    
    # Run main function
    matches, video_angle = main(args.video, args.csv, args.output, args.fps)
    
    if matches:
        print(f"\n{'='*60}")
        print(f"SUCCESS: Detected {len(matches)} matches")
        print(f"{'='*60}")
        for i, (start, end) in enumerate(matches, 1):
            print(f"Match {i}: Frames {start} to {end} (Duration: {end - start + 1} frames)")
        print(f"Video angle: {video_angle}")
        print(f"\nResults saved to: {args.output}")
    else:
        print(f"\n{'='*60}")
        print(f"WARNING: No matches detected")
        print(f"{'='*60}")
        exit(1)
