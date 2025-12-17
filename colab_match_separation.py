#!/usr/bin/env python3
"""
Colab Match Separation Script - Simplified for debugging

This script processes keypoint data to detect fencing match start and end frames.
No file outputs - just returns the data for debugging.

Usage in Colab:
    from colab_match_separation import detect_start_frames_from_csv
    
    start_frames = detect_start_frames_from_csv(
        left_xdata_path='path/to/left_xdata.csv',
        left_ydata_path='path/to/left_ydata.csv',
        right_xdata_path='path/to/right_xdata.csv',
        right_ydata_path='path/to/right_ydata.csv'
    )
"""

import numpy as np
import pandas as pd
import logging

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s'
)
logger = logging.getLogger()

def extract_keypoints(xdata, ydata, index):
    """Extract keypoints at a specific index"""
    keypoints = {
        'hipback': (xdata.loc[index, '11'], ydata.loc[index, '11']),
        'kneeback': (xdata.loc[index, '13'], ydata.loc[index, '13']),
        'ankleback': (xdata.loc[index, '15'], ydata.loc[index, '15']),
        'hipfront': (xdata.loc[index, '12'], ydata.loc[index, '12']),
        'kneefront': (xdata.loc[index, '14'], ydata.loc[index, '14']),
        'anklefront': (xdata.loc[index, '16'], ydata.loc[index, '16']),
        'handfront': (xdata.loc[index, '10'], ydata.loc[index, '10'])
    }
    return keypoints

def calculate_angle(p1, p2, p3):
    """Calculate angle between three points"""
    a = np.array(p1)
    b = np.array(p2)
    c = np.array(p3)
    ab = a - b
    cb = c - b
    norm_ab = np.linalg.norm(ab)
    norm_cb = np.linalg.norm(cb)
    if norm_ab == 0 or norm_cb == 0:
        return 0.0
    cosine_angle = np.dot(ab, cb) / (norm_ab * norm_cb)
    angle = np.degrees(np.arccos(np.clip(cosine_angle, -1.0, 1.0)))
    return angle

def calculate_fencer_angles(fencer_keypoints):
    """Calculate body angles for a fencer"""
    angles = {
        'back_knee_angle': calculate_angle(fencer_keypoints['ankleback'], fencer_keypoints['kneeback'], fencer_keypoints['hipback']),
        'back_hip_angle': calculate_angle(fencer_keypoints['kneeback'], fencer_keypoints['hipback'], fencer_keypoints['hipfront']),
        'front_hip_angle': calculate_angle(fencer_keypoints['hipback'], fencer_keypoints['hipfront'], fencer_keypoints['kneefront']),
        'front_knee_angle': calculate_angle(fencer_keypoints['hipfront'], fencer_keypoints['kneefront'], fencer_keypoints['anklefront']),
        'hand_hip_angle': calculate_angle(fencer_keypoints['handfront'], fencer_keypoints['hipfront'], fencer_keypoints['kneefront']),
    }
    return angles

def detect_start_frames_from_csv(left_xdata_path, left_ydata_path, right_xdata_path, right_ydata_path, 
                                  movement_threshold=0.01, static_threshold=0.01, min_static_frames=5, 
                                  min_moving_frames=5, static_window=5, lookahead_frames=10, 
                                  min_frame_spacing=120, angle_diff_threshold=180, distance_tolerance=0.2):
    """
    Detect start frames from CSV files.
    
    Args:
        left_xdata_path: Path to left fencer X coordinates CSV
        left_ydata_path: Path to left fencer Y coordinates CSV
        right_xdata_path: Path to right fencer X coordinates CSV
        right_ydata_path: Path to right fencer Y coordinates CSV
        (other parameters are tuning parameters for detection)
    
    Returns:
        List of start frame numbers
    """
    # Load CSV files
    left_xdata_df = pd.read_csv(left_xdata_path, dtype={str(i): float for i in range(7, 17)})
    left_ydata_df = pd.read_csv(left_ydata_path, dtype={str(i): float for i in range(7, 17)})
    right_xdata_df = pd.read_csv(right_xdata_path, dtype={str(i): float for i in range(7, 17)})
    right_ydata_df = pd.read_csv(right_ydata_path, dtype={str(i): float for i in range(7, 17)})
    
    # Reset index and add Frame column
    left_xdata_df = left_xdata_df.reset_index(drop=True)
    left_ydata_df = left_ydata_df.reset_index(drop=True)
    right_xdata_df = right_xdata_df.reset_index(drop=True)
    right_ydata_df = right_ydata_df.reset_index(drop=True)
    
    left_xdata_df['Frame'] = range(len(left_xdata_df))
    left_ydata_df['Frame'] = range(len(left_ydata_df))
    right_xdata_df['Frame'] = range(len(right_xdata_df))
    right_ydata_df['Frame'] = range(len(right_ydata_df))
    
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
        logger.error(f"Error extracting reference pose from frame {ref_frame}: {e}")
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


def detect_start_frames_from_dataframes(left_xdata_df, left_ydata_df, right_xdata_df, right_ydata_df,
                                         movement_threshold=0.01, static_threshold=0.01, min_static_frames=5,
                                         min_moving_frames=5, static_window=5, lookahead_frames=10,
                                         min_frame_spacing=120, angle_diff_threshold=180, distance_tolerance=0.2):
    """
    Detect start frames from DataFrames (if you already have them loaded).
    
    Args:
        left_xdata_df: DataFrame with left fencer X coordinates
        left_ydata_df: DataFrame with left fencer Y coordinates
        right_xdata_df: DataFrame with right fencer X coordinates
        right_ydata_df: DataFrame with right fencer Y coordinates
        (other parameters are tuning parameters for detection)
    
    Returns:
        List of start frame numbers
    """
    # Reset index and add Frame column if not present
    if 'Frame' not in left_xdata_df.columns:
        left_xdata_df = left_xdata_df.reset_index(drop=True)
        left_ydata_df = left_ydata_df.reset_index(drop=True)
        right_xdata_df = right_xdata_df.reset_index(drop=True)
        right_ydata_df = right_ydata_df.reset_index(drop=True)
        
        left_xdata_df['Frame'] = range(len(left_xdata_df))
        left_ydata_df['Frame'] = range(len(left_ydata_df))
        right_xdata_df['Frame'] = range(len(right_xdata_df))
        right_ydata_df['Frame'] = range(len(right_ydata_df))
    
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
        logger.error(f"Error extracting reference pose from frame {ref_frame}: {e}")
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
