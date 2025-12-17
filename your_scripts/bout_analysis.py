import numpy as np
import pandas as pd
import os
import json
from scipy.signal import savgol_filter
import google.generativeai as genai
import logging
import traceback
from models import Bout
import math
from your_scripts.bout_classification import classify_bout_categories

import numpy as np
import pandas as pd
from scipy.signal import find_peaks, savgol_filter
from scipy.ndimage import gaussian_filter1d
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Optional

def calculate_velocity_acceleration(positions, fps=30):
    """
    Calculate velocity and acceleration from position data.
    
    Parameters:
    - positions: List of positions over time
    - fps: Frames per second
    
    Returns:
    - velocities: List of velocities
    - accelerations: List of accelerations
    """
    positions = np.array(positions)
    dt = 1.0 / fps  # Time per frame
    
    # Calculate velocity (first derivative)
    velocities = np.gradient(positions) / dt
    
    # Calculate acceleration (second derivative)
    accelerations = np.gradient(velocities) / dt
    
    return velocities.tolist(), accelerations.tolist()

def merge_nearby_launch_frames(launch_frames, min_gap=2, min_duration=5):
    """
    Merge launch frames that are close together and filter by minimum duration.
    
    Parameters:
    - launch_frames: List of frame numbers
    - min_gap: Minimum gap between separate launches (frames)
    - min_duration: Minimum duration for a valid launch (frames)
    
    Returns:
    - List of tuples (start_frame, end_frame) for valid launches
    """
    if not launch_frames:
        return []
    
    launch_frames = sorted(launch_frames)
    merged_intervals = []
    current_start = launch_frames[0]
    current_end = launch_frames[0]
    
    for frame in launch_frames[1:]:
        if frame - current_end <= min_gap:
            # Extend current interval
            current_end = frame
        else:
            # Check if current interval is long enough
            if current_end - current_start + 1 >= min_duration:
                merged_intervals.append((current_start, current_end))
            # Start new interval
            current_start = frame
            current_end = frame
    
    # Don't forget the last interval
    if current_end - current_start + 1 >= min_duration:
        merged_intervals.append((current_start, current_end))
    
    return merged_intervals

def extract_launch_metrics(xdata_df, ydata_df, launch_interval, fps=30):
    """
    Extract detailed metrics for a launch interval.
    
    Parameters:
    - xdata_df, ydata_df: Fencer keypoint data
    - launch_interval: Tuple (start_frame, end_frame)
    - fps: Frames per second
    
    Returns:
    - Dictionary with launch metrics
    """
    start_frame, end_frame = launch_interval
    
    # Filter dataframe for the launch frames
    launch_mask = (xdata_df['Frame'] >= start_frame) & (xdata_df['Frame'] <= end_frame)
    launch_xdata = xdata_df[launch_mask]
    launch_ydata = ydata_df[launch_mask]
    
    if len(launch_xdata) == 0:
        return None
    
    # Extract keypoint data for the launch period
    # Front foot (keypoint 16)
    front_foot_x = launch_xdata['16'].values
    front_foot_y = launch_ydata['16'].values
    
    # Back foot (keypoint 15)
    back_foot_x = launch_xdata['15'].values
    back_foot_y = launch_ydata['15'].values
    
    # Front hip (assuming keypoint 12 is right hip, which would be front for right fencer)
    # You may need to adjust this based on your fencer orientation
    front_hip_x = launch_xdata['12'].values
    front_hip_y = launch_ydata['12'].values
    
    # Calculate velocities and accelerations
    front_foot_vel_x, front_foot_acc_x = calculate_velocity_acceleration(front_foot_x, fps)
    front_foot_vel_y, front_foot_acc_y = calculate_velocity_acceleration(front_foot_y, fps)
    
    back_foot_vel_x, back_foot_acc_x = calculate_velocity_acceleration(back_foot_x, fps)
    back_foot_vel_y, back_foot_acc_y = calculate_velocity_acceleration(back_foot_y, fps)
    
    front_hip_vel_x, front_hip_acc_x = calculate_velocity_acceleration(front_hip_x, fps)
    front_hip_vel_y, front_hip_acc_y = calculate_velocity_acceleration(front_hip_y, fps)
    
    # Calculate magnitude velocities and accelerations
    front_foot_vel_mag = [math.sqrt(vx**2 + vy**2) for vx, vy in zip(front_foot_vel_x, front_foot_vel_y)]
    front_foot_acc_mag = [math.sqrt(ax**2 + ay**2) for ax, ay in zip(front_foot_acc_x, front_foot_acc_y)]
    
    back_foot_vel_mag = [math.sqrt(vx**2 + vy**2) for vx, vy in zip(back_foot_vel_x, back_foot_vel_y)]
    back_foot_acc_mag = [math.sqrt(ax**2 + ay**2) for ax, ay in zip(back_foot_acc_x, back_foot_acc_y)]
    
    front_hip_vel_mag = [math.sqrt(vx**2 + vy**2) for vx, vy in zip(front_hip_vel_x, front_hip_vel_y)]
    front_hip_acc_mag = [math.sqrt(ax**2 + ay**2) for ax, ay in zip(front_hip_acc_x, front_hip_acc_y)]
    
    # Calculate foot separation over time
    foot_distances = [math.sqrt((fx - bx)**2 + (fy - by)**2) 
                     for fx, bx, fy, by in zip(front_foot_x, back_foot_x, front_foot_y, back_foot_y)]
    
    # Derived metrics
    duration_frames = end_frame - start_frame + 1
    duration_seconds = duration_frames / fps
    
    # Peak values
    max_front_foot_velocity = max(front_foot_vel_mag) if front_foot_vel_mag else 0
    max_front_foot_acceleration = max(front_foot_acc_mag) if front_foot_acc_mag else 0
    max_back_foot_velocity = max(back_foot_vel_mag) if back_foot_vel_mag else 0
    max_back_foot_acceleration = max(back_foot_acc_mag) if back_foot_acc_mag else 0
    max_front_hip_velocity = max(front_hip_vel_mag) if front_hip_vel_mag else 0
    max_front_hip_acceleration = max(front_hip_acc_mag) if front_hip_acc_mag else 0
    
    # Average values
    avg_front_foot_velocity = np.mean(front_foot_vel_mag) if front_foot_vel_mag else 0
    avg_front_foot_acceleration = np.mean(front_foot_acc_mag) if front_foot_acc_mag else 0
    avg_back_foot_velocity = np.mean(back_foot_vel_mag) if back_foot_vel_mag else 0
    avg_back_foot_acceleration = np.mean(back_foot_acc_mag) if back_foot_acc_mag else 0
    avg_front_hip_velocity = np.mean(front_hip_vel_mag) if front_hip_vel_mag else 0
    avg_front_hip_acceleration = np.mean(front_hip_acc_mag) if front_hip_acc_mag else 0
    
    # Foot separation metrics
    initial_foot_distance = foot_distances[0] if foot_distances else 0
    max_foot_distance = max(foot_distances) if foot_distances else 0
    final_foot_distance = foot_distances[-1] if foot_distances else 0
    foot_extension_ratio = max_foot_distance / initial_foot_distance if initial_foot_distance > 0 else 1
    
    # Promptness - how quickly maximum extension is reached
    max_distance_frame_idx = foot_distances.index(max_foot_distance) if foot_distances else 0
    promptness_ratio = max_distance_frame_idx / len(foot_distances) if foot_distances else 0.5
    
    metrics = {
        'start_frame': start_frame,
        'end_frame': end_frame,
        'duration_frames': duration_frames,
        'duration_seconds': duration_seconds,
        
        # Front foot metrics
        'front_foot_max_velocity': max_front_foot_velocity,
        'front_foot_avg_velocity': avg_front_foot_velocity,
        'front_foot_max_acceleration': max_front_foot_acceleration,
        'front_foot_avg_acceleration': avg_front_foot_acceleration,
        'front_foot_velocity_profile': front_foot_vel_mag,
        'front_foot_acceleration_profile': front_foot_acc_mag,
        
        # Back foot metrics
        'back_foot_max_velocity': max_back_foot_velocity,
        'back_foot_avg_velocity': avg_back_foot_velocity,
        'back_foot_max_acceleration': max_back_foot_acceleration,
        'back_foot_avg_acceleration': avg_back_foot_acceleration,
        'back_foot_velocity_profile': back_foot_vel_mag,
        'back_foot_acceleration_profile': back_foot_acc_mag,
        
        # Front hip metrics
        'front_hip_max_velocity': max_front_hip_velocity,
        'front_hip_avg_velocity': avg_front_hip_velocity,
        'front_hip_max_acceleration': max_front_hip_acceleration,
        'front_hip_avg_acceleration': avg_front_hip_acceleration,
        'front_hip_velocity_profile': front_hip_vel_mag,
        'front_hip_acceleration_profile': front_hip_acc_mag,
        
        # Foot distance metrics
        'initial_foot_distance': initial_foot_distance,
        'max_foot_distance': max_foot_distance,
        'final_foot_distance': final_foot_distance,
        'foot_extension_ratio': foot_extension_ratio,
        'foot_distance_profile': foot_distances,
        
        # Derived metrics
        'promptness_ratio': promptness_ratio,  # 0 = immediate max, 1 = delayed max
        'launch_intensity': max_front_foot_velocity * foot_extension_ratio,  # Combined metric
    }
    
    return metrics

def detect_launches_in_advance_intervals(xdata_df, ydata_df, advance_intervals, fps=30, 
                                       threshold_distance=1, min_gap=2, min_duration=3):
    """
    Detect launches within advance intervals and extract detailed metrics.
    
    Parameters:
    - xdata_df, ydata_df: Fencer keypoint data
    - advance_intervals: List of (start_frame, end_frame) for advance intervals
    - fps: Frames per second
    - threshold_distance: Hard-coded threshold for launch detection
    - min_gap: Minimum gap between separate launches (frames)
    - min_duration: Minimum duration for a valid launch (frames)
    
    Returns:
    - List of launch metrics dictionaries
    """
    all_launches = []
    
    for interval_idx, (start_frame, end_frame) in enumerate(advance_intervals):
        
        # Filter data for this advance interval
        interval_mask = (xdata_df['Frame'] >= start_frame) & (xdata_df['Frame'] <= end_frame)
        interval_xdata = xdata_df[interval_mask]
        
        if len(interval_xdata) == 0:
            continue
        
        # Detect launch frames within this interval
        launch_frames = []
        
        for _, row in interval_xdata.iterrows():
            front_foot_x = row['16']
            back_foot_x = row['15']
            
            current_distance = abs(front_foot_x - back_foot_x)  # Using x-distance only for simplicity
            
            if current_distance > threshold_distance:
                launch_frames.append(row['Frame'])
        
        if launch_frames:
            # Merge nearby launch frames
            merged_launches = merge_nearby_launch_frames(launch_frames, min_gap, min_duration)
            
            # Extract metrics for each valid launch
            # Only keep the last launch from each interval
            if merged_launches:
                launch_interval = merged_launches[-1]  # Take the last launch
                metrics = extract_launch_metrics(xdata_df, ydata_df, launch_interval, fps)
                if metrics:
                    metrics['advance_interval_index'] = interval_idx
                    metrics['advance_interval_frames'] = (start_frame, end_frame)
                    all_launches.append(metrics)
    
    return all_launches

def analyze_launches_for_both_fencers(left_xdata_df, left_ydata_df, right_xdata_df, right_ydata_df,
                                    left_advance_intervals, right_advance_intervals, fps=30):
    """
    Analyze launches for both fencers within their advance intervals.
    
    Parameters:
    - left_xdata_df, left_ydata_df: Left fencer data
    - right_xdata_df, right_ydata_df: Right fencer data
    - left_advance_intervals: Left fencer advance intervals (frame numbers)
    - right_advance_intervals: Right fencer advance intervals (frame numbers)
    - fps: Frames per second
    
    Returns:
    - Dictionary with launch analyses for both fencers
    """
    
    left_launches = detect_launches_in_advance_intervals(
        left_xdata_df, left_ydata_df, left_advance_intervals, fps
    )
    
    right_launches = detect_launches_in_advance_intervals(
        right_xdata_df, right_ydata_df, right_advance_intervals, fps
    )
    
    # Summary statistics
    left_summary = {
        'total_launches': len(left_launches),
        'total_advance_intervals': len(left_advance_intervals),
        'launches_per_advance': len(left_launches) / len(left_advance_intervals) if left_advance_intervals else 0,
        'avg_launch_duration': np.mean([l['duration_seconds'] for l in left_launches]) if left_launches else 0,
        'max_launch_intensity': max([l['launch_intensity'] for l in left_launches]) if left_launches else 0,
    }
    
    right_summary = {
        'total_launches': len(right_launches),
        'total_advance_intervals': len(right_advance_intervals),
        'launches_per_advance': len(right_launches) / len(right_advance_intervals) if right_advance_intervals else 0,
        'avg_launch_duration': np.mean([l['duration_seconds'] for l in right_launches]) if right_launches else 0,
        'max_launch_intensity': max([l['launch_intensity'] for l in right_launches]) if right_launches else 0,
    }
    
    return {
        'left_fencer': {
            'launches': left_launches,
            'summary': left_summary
        },
        'right_fencer': {
            'launches': right_launches,
            'summary': right_summary
        }
    }

def merge_nearby_extension_frames(extension_frames, min_gap=2, min_duration=5):
    """
    Merge extension frames that are close together and filter by minimum duration.
    
    Parameters:
    - extension_frames: List of frame numbers
    - min_gap: Minimum gap between separate extensions (frames)
    - min_duration: Minimum duration for a valid extension (frames)
    
    Returns:
    - List of tuples (start_frame, end_frame) for valid extensions
    """
    if not extension_frames:
        return []
    
    extension_frames = sorted(extension_frames)
    merged_intervals = []
    current_start = extension_frames[0]
    current_end = extension_frames[0]
    
    for frame in extension_frames[1:]:
        if frame - current_end <= min_gap:
            # Extend current interval
            current_end = frame
        else:
            # Check if current interval is long enough
            if current_end - current_start + 1 >= min_duration:
                merged_intervals.append((current_start, current_end))
            # Start new interval
            current_start = frame
            current_end = frame
    
    # Don't forget the last interval
    if current_end - current_start + 1 >= min_duration:
        merged_intervals.append((current_start, current_end))
    
    return merged_intervals

def extract_extension_metrics(xdata_df, ydata_df, extension_interval, fps=30):
    """
    Extract detailed metrics for an arm extension interval.
    
    Parameters:
    - xdata_df, ydata_df: Fencer keypoint data
    - extension_interval: Tuple (start_frame, end_frame)
    - fps: Frames per second
    
    Returns:
    - Dictionary with extension metrics
    """
    start_frame, end_frame = extension_interval
    
    # Filter dataframe for the extension frames
    extension_mask = (xdata_df['Frame'] >= start_frame) & (xdata_df['Frame'] <= end_frame)
    extension_xdata = xdata_df[extension_mask]
    extension_ydata = ydata_df[extension_mask]
    
    if len(extension_xdata) == 0:
        return None
    
    # Extract keypoint data for the extension period
    # Hand (keypoint 10 - right wrist)
    hand_x = extension_xdata['10'].values
    hand_y = extension_ydata['10'].values
    
    # Hip (keypoint 12 - right hip)
    hip_x = extension_xdata['12'].values
    hip_y = extension_ydata['12'].values
    
    # Elbow (keypoint 8 - right elbow)
    elbow_x = extension_xdata['8'].values
    elbow_y = extension_ydata['8'].values
    
    # Shoulder (keypoint 6 - right shoulder)
    shoulder_x = extension_xdata['6'].values
    shoulder_y = extension_ydata['6'].values
    
    # Calculate velocities and accelerations for hand
    hand_vel_x, hand_acc_x = calculate_velocity_acceleration(hand_x, fps)
    hand_vel_y, hand_acc_y = calculate_velocity_acceleration(hand_y, fps)
    
    # Calculate velocities and accelerations for elbow
    elbow_vel_x, elbow_acc_x = calculate_velocity_acceleration(elbow_x, fps)
    elbow_vel_y, elbow_acc_y = calculate_velocity_acceleration(elbow_y, fps)
    
    # Calculate velocities and accelerations for shoulder
    shoulder_vel_x, shoulder_acc_x = calculate_velocity_acceleration(shoulder_x, fps)
    shoulder_vel_y, shoulder_acc_y = calculate_velocity_acceleration(shoulder_y, fps)
    
    # Calculate magnitude velocities and accelerations
    hand_vel_mag = [math.sqrt(vx**2 + vy**2) for vx, vy in zip(hand_vel_x, hand_vel_y)]
    hand_acc_mag = [math.sqrt(ax**2 + ay**2) for ax, ay in zip(hand_acc_x, hand_acc_y)]
    
    elbow_vel_mag = [math.sqrt(vx**2 + vy**2) for vx, vy in zip(elbow_vel_x, elbow_vel_y)]
    elbow_acc_mag = [math.sqrt(ax**2 + ay**2) for ax, ay in zip(elbow_acc_x, elbow_acc_y)]
    
    shoulder_vel_mag = [math.sqrt(vx**2 + vy**2) for vx, vy in zip(shoulder_vel_x, shoulder_vel_y)]
    shoulder_acc_mag = [math.sqrt(ax**2 + ay**2) for ax, ay in zip(shoulder_acc_x, shoulder_acc_y)]
    
    # Calculate hand-to-hip distances over time
    hand_hip_distances = [math.sqrt((hx - hipx)**2 + (hy - hipy)**2) 
                         for hx, hipx, hy, hipy in zip(hand_x, hip_x, hand_y, hip_y)]
    
    # Calculate arm angles (elbow angle)
    arm_angles = []
    for i in range(len(hand_x)):
        # Vector from elbow to shoulder
        shoulder_vec = (shoulder_x[i] - elbow_x[i], shoulder_y[i] - elbow_y[i])
        # Vector from elbow to hand
        hand_vec = (hand_x[i] - elbow_x[i], hand_y[i] - elbow_y[i])
        
        # Calculate angle between vectors
        dot_product = shoulder_vec[0] * hand_vec[0] + shoulder_vec[1] * hand_vec[1]
        shoulder_mag = math.sqrt(shoulder_vec[0]**2 + shoulder_vec[1]**2)
        hand_mag = math.sqrt(hand_vec[0]**2 + hand_vec[1]**2)
        
        if shoulder_mag > 0 and hand_mag > 0:
            cos_angle = dot_product / (shoulder_mag * hand_mag)
            cos_angle = max(-1, min(1, cos_angle))  # Clamp to [-1, 1]
            angle = math.acos(cos_angle) * 180 / math.pi  # Convert to degrees
            arm_angles.append(angle)
        else:
            arm_angles.append(0)
    
    # Derived metrics
    duration_frames = end_frame - start_frame + 1
    duration_seconds = duration_frames / fps
    
    # Peak values
    max_hand_velocity = max(hand_vel_mag) if hand_vel_mag else 0
    max_hand_acceleration = max(hand_acc_mag) if hand_acc_mag else 0
    max_elbow_velocity = max(elbow_vel_mag) if elbow_vel_mag else 0
    max_elbow_acceleration = max(elbow_acc_mag) if elbow_acc_mag else 0
    max_shoulder_velocity = max(shoulder_vel_mag) if shoulder_vel_mag else 0
    max_shoulder_acceleration = max(shoulder_acc_mag) if shoulder_acc_mag else 0
    
    # Average values
    avg_hand_velocity = np.mean(hand_vel_mag) if hand_vel_mag else 0
    avg_hand_acceleration = np.mean(hand_acc_mag) if hand_acc_mag else 0
    avg_elbow_velocity = np.mean(elbow_vel_mag) if elbow_vel_mag else 0
    avg_elbow_acceleration = np.mean(elbow_acc_mag) if elbow_acc_mag else 0
    avg_shoulder_velocity = np.mean(shoulder_vel_mag) if shoulder_vel_mag else 0
    avg_shoulder_acceleration = np.mean(shoulder_acc_mag) if shoulder_acc_mag else 0
    
    # Extension metrics
    initial_hand_hip_distance = hand_hip_distances[0] if hand_hip_distances else 0
    max_hand_hip_distance = max(hand_hip_distances) if hand_hip_distances else 0
    final_hand_hip_distance = hand_hip_distances[-1] if hand_hip_distances else 0
    extension_ratio = max_hand_hip_distance / initial_hand_hip_distance if initial_hand_hip_distance > 0 else 1
    
    # Arm angle metrics
    initial_arm_angle = arm_angles[0] if arm_angles else 0
    max_arm_angle = max(arm_angles) if arm_angles else 0
    final_arm_angle = arm_angles[-1] if arm_angles else 0
    angle_change = final_arm_angle - initial_arm_angle
    
    # Extension speed - how quickly maximum extension is reached
    max_distance_frame_idx = hand_hip_distances.index(max_hand_hip_distance) if hand_hip_distances else 0
    extension_speed_ratio = max_distance_frame_idx / len(hand_hip_distances) if hand_hip_distances else 0.5
    
    # Extension frequency within this period (changes in direction)
    extension_frequency = 0
    if len(hand_hip_distances) > 2:
        distances_diff = np.diff(hand_hip_distances)
        sign_changes = np.diff(np.sign(distances_diff))
        extension_frequency = np.count_nonzero(sign_changes) / duration_seconds if duration_seconds > 0 else 0
    
    metrics = {
        'start_frame': start_frame,
        'end_frame': end_frame,
        'duration_frames': duration_frames,
        'duration_seconds': duration_seconds,
        
        # Hand metrics
        'hand_max_velocity': max_hand_velocity,
        'hand_avg_velocity': avg_hand_velocity,
        'hand_max_acceleration': max_hand_acceleration,
        'hand_avg_acceleration': avg_hand_acceleration,
        'hand_velocity_profile': hand_vel_mag,
        'hand_acceleration_profile': hand_acc_mag,
        
        # Elbow metrics
        'elbow_max_velocity': max_elbow_velocity,
        'elbow_avg_velocity': avg_elbow_velocity,
        'elbow_max_acceleration': max_elbow_acceleration,
        'elbow_avg_acceleration': avg_elbow_acceleration,
        'elbow_velocity_profile': elbow_vel_mag,
        'elbow_acceleration_profile': elbow_acc_mag,
        
        # Shoulder metrics
        'shoulder_max_velocity': max_shoulder_velocity,
        'shoulder_avg_velocity': avg_shoulder_velocity,
        'shoulder_max_acceleration': max_shoulder_acceleration,
        'shoulder_avg_acceleration': avg_shoulder_acceleration,
        'shoulder_velocity_profile': shoulder_vel_mag,
        'shoulder_acceleration_profile': shoulder_acc_mag,
        
        # Extension distance metrics
        'initial_hand_hip_distance': initial_hand_hip_distance,
        'max_hand_hip_distance': max_hand_hip_distance,
        'final_hand_hip_distance': final_hand_hip_distance,
        'extension_ratio': extension_ratio,
        'hand_hip_distance_profile': hand_hip_distances,
        
        # Arm angle metrics
        'initial_arm_angle': initial_arm_angle,
        'max_arm_angle': max_arm_angle,
        'final_arm_angle': final_arm_angle,
        'angle_change': angle_change,
        'arm_angle_profile': arm_angles,
        
        # Derived metrics
        'extension_speed_ratio': extension_speed_ratio,  # 0 = immediate max, 1 = delayed max
        'extension_frequency': extension_frequency,  # Changes per second
        'extension_intensity': max_hand_velocity * extension_ratio,  # Combined metric
        'arm_coordination': max_hand_velocity / max_elbow_velocity if max_elbow_velocity > 0 else 1,  # Hand vs elbow speed ratio
    }
    
    return metrics

def detect_arm_extensions_in_advance_intervals(xdata_df, ydata_df, advance_intervals, fps=30,
                                             threshold_distance=0.35, min_gap=2, min_duration=5):
    """
    Detect arm extensions within advance intervals and extract detailed metrics.
    
    Parameters:
    - xdata_df, ydata_df: Fencer keypoint data
    - advance_intervals: List of (start_frame, end_frame) for advance intervals
    - fps: Frames per second
    - threshold_distance: Hard-coded threshold for extension detection (pixels)
    - min_gap: Minimum gap between separate extensions (frames)
    - min_duration: Minimum duration for a valid extension (frames)
    
    Returns:
    - List of extension metrics dictionaries
    """
    all_extensions = []
    
    for interval_idx, (start_frame, end_frame) in enumerate(advance_intervals):
        # Filter data for this advance interval
        interval_mask = (xdata_df['Frame'] >= start_frame) & (xdata_df['Frame'] <= end_frame)
        interval_xdata = xdata_df[interval_mask]
        interval_ydata = ydata_df[interval_mask]
        
        if len(interval_xdata) == 0:
            continue
        
        # Detect extension frames within this interval
        extension_frames = []
        
        for _, row in interval_xdata.iterrows():
            # Use right wrist (keypoint 10) and right hip (keypoint 12) for arm extension
            hand_x = row['10']  # Right wrist
            hand_y = interval_ydata.loc[row.name, '10']
            hip_x = row['12']   # Right hip  
            hip_y = interval_ydata.loc[row.name, '12']
            
            # Calculate distance between hand and hip
            current_distance = math.sqrt((hand_x - hip_x)**2 + (hand_y - hip_y)**2)
            
            if current_distance > threshold_distance:
                extension_frames.append(row['Frame'])
        
        if extension_frames:
            # Merge nearby extension frames
            merged_extensions = merge_nearby_extension_frames(extension_frames, min_gap, min_duration)
            
            # Only keep the last extension from each interval
            if merged_extensions:
                extension_interval = merged_extensions[-1]  # Take the last extension
                metrics = extract_extension_metrics(xdata_df, ydata_df, extension_interval, fps)
                if metrics:
                    metrics['advance_interval_index'] = interval_idx
                    metrics['advance_interval_frames'] = (start_frame, end_frame)
                    metrics['threshold_used'] = threshold_distance
                    all_extensions.append(metrics)
    
    return all_extensions

def analyze_arm_extensions_for_both_fencers(left_xdata_df, left_ydata_df, right_xdata_df, right_ydata_df,
                                          left_advance_intervals, right_advance_intervals, fps=30,
                                          threshold_distance=0.35):
    """
    Analyze arm extensions for both fencers within their advance intervals.
    
    Parameters:
    - left_xdata_df, left_ydata_df: Left fencer data
    - right_xdata_df, right_ydata_df: Right fencer data
    - left_advance_intervals: Left fencer advance intervals (frame numbers)
    - right_advance_intervals: Right fencer advance intervals (frame numbers)
    - fps: Frames per second
    - threshold_distance: Hard-coded threshold for extension detection
    
    Returns:
    - Dictionary with extension analyses for both fencers
    """
    
    left_extensions = detect_arm_extensions_in_advance_intervals(
        left_xdata_df, left_ydata_df, left_advance_intervals, fps, threshold_distance
    )
    
    right_extensions = detect_arm_extensions_in_advance_intervals(
        right_xdata_df, right_ydata_df, right_advance_intervals, fps, threshold_distance
    )
    
    # Comprehensive summary statistics
    left_summary = {
        'total_extensions': len(left_extensions),
        'total_advance_intervals': len(left_advance_intervals),
        'extensions_per_advance': len(left_extensions) / len(left_advance_intervals) if left_advance_intervals else 0,
        'avg_extension_duration': np.mean([e['duration_seconds'] for e in left_extensions]) if left_extensions else 0,
        'max_hand_velocity': max([e['hand_max_velocity'] for e in left_extensions]) if left_extensions else 0,
        'avg_hand_velocity': np.mean([e['hand_avg_velocity'] for e in left_extensions]) if left_extensions else 0,
        'max_extension_ratio': max([e['extension_ratio'] for e in left_extensions]) if left_extensions else 0,
        'avg_extension_frequency': np.mean([e['extension_frequency'] for e in left_extensions]) if left_extensions else 0,
        'max_extension_intensity': max([e['extension_intensity'] for e in left_extensions]) if left_extensions else 0,
    }
    
    right_summary = {
        'total_extensions': len(right_extensions),
        'total_advance_intervals': len(right_advance_intervals),
        'extensions_per_advance': len(right_extensions) / len(right_advance_intervals) if right_advance_intervals else 0,
        'avg_extension_duration': np.mean([e['duration_seconds'] for e in right_extensions]) if right_extensions else 0,
        'max_hand_velocity': max([e['hand_max_velocity'] for e in right_extensions]) if right_extensions else 0,
        'avg_hand_velocity': np.mean([e['hand_avg_velocity'] for e in right_extensions]) if right_extensions else 0,
        'max_extension_ratio': max([e['extension_ratio'] for e in right_extensions]) if right_extensions else 0,
        'avg_extension_frequency': np.mean([e['extension_frequency'] for e in right_extensions]) if right_extensions else 0,
        'max_extension_intensity': max([e['extension_intensity'] for e in right_extensions]) if right_extensions else 0,
    }
    
    return {
        'left_fencer': {
            'extensions': left_extensions,
            'summary': left_summary
        },
        'right_fencer': {
            'extensions': right_extensions,
            'summary': right_summary
        }
    }


class FencingMovementDetector:
    def __init__(self, velocity_threshold=0.03, min_interval_length=7, smoothing_window=3):
        """
        Initialize the movement detector with configurable parameters
        
        Parameters:
        - velocity_threshold: Threshold for distinguishing movement vs pause
        - min_interval_length: Minimum frames for a valid interval
        - smoothing_window: Window size for smoothing (must be odd)
        """
        self.velocity_threshold = velocity_threshold
        self.min_interval_length = min_interval_length
        self.smoothing_window = smoothing_window if smoothing_window % 2 == 1 else smoothing_window + 1
    
    def extract_position_features(self, xdata_df, is_left_fencer=True):
        """
        Extract and compute position-based features for movement analysis
        
        Parameters:
        - xdata_df: DataFrame with keypoint x-coordinates
        - is_left_fencer: Boolean indicating fencer side
        
        Returns:
        - Dict with computed features
        """
        # Extract frame numbers - FIXED: Store real frame indices
        if 'Frame' in xdata_df.columns:
            frame_indices = xdata_df['Frame'].values
        else:
            frame_indices = np.arange(len(xdata_df))
        
        # Extract keypoints (assuming standard pose keypoint indices)
        try:
            left_hip = xdata_df['11'].values if '11' in xdata_df.columns else np.zeros(len(xdata_df))
            right_hip = xdata_df['12'].values if '12' in xdata_df.columns else np.zeros(len(xdata_df))
            left_ankle = xdata_df['15'].values if '15' in xdata_df.columns else np.zeros(len(xdata_df))
            right_ankle = xdata_df['16'].values if '16' in xdata_df.columns else np.zeros(len(xdata_df))
            
            # Additional keypoints for better analysis
            left_knee = xdata_df['13'].values if '13' in xdata_df.columns else np.zeros(len(xdata_df))
            right_knee = xdata_df['14'].values if '14' in xdata_df.columns else np.zeros(len(xdata_df))
        except KeyError as e:
            print(f"Missing keypoint column: {e}")
            print(f"Available columns: {list(xdata_df.columns)}")
            return None
        
        # Handle missing data by interpolation
        def interpolate_missing(data):
            mask = (data != 0) & (~np.isnan(data))
            if np.sum(mask) < 2:
                return data
            indices = np.arange(len(data))
            data_interp = np.interp(indices, indices[mask], data[mask])
            return data_interp
        
        left_hip = interpolate_missing(left_hip)
        right_hip = interpolate_missing(right_hip)
        left_ankle = interpolate_missing(left_ankle)
        right_ankle = interpolate_missing(right_ankle)
        left_knee = interpolate_missing(left_knee)
        right_knee = interpolate_missing(right_knee)
        
        # Compute center of mass and key positions
        hip_center = (left_hip + right_hip) / 2
        ankle_center = (left_ankle + right_ankle) / 2
        knee_center = (left_knee + right_knee) / 2
        
        # Determine front and back foot based on fencer orientation
        if is_left_fencer:
            # For left fencer, assume right foot is front, left foot is back
            front_foot = right_ankle
            back_foot = left_ankle
            front_knee = right_knee
            back_knee = left_knee
        else:
            # For right fencer, assume left foot is front, right foot is back
            front_foot = left_ankle
            back_foot = right_ankle
            front_knee = left_knee
            back_knee = right_knee
        
        # Create weighted position combining multiple keypoints
        # More weight on front foot and hip center for advance detection
        weighted_position = 0.5 * front_foot + 0.3 * hip_center + 0.2 * front_knee
        
        return {
            'frame_indices': frame_indices,  # FIXED: Include real frame indices
            'hip_center': hip_center,
            'ankle_center': ankle_center,
            'knee_center': knee_center,
            'front_foot': front_foot,
            'back_foot': back_foot,
            'front_knee': front_knee,
            'back_knee': back_knee,
            'weighted_position': weighted_position,
            'is_left_fencer': is_left_fencer
        }
    
    def smooth_signals(self, features):
        """Apply smoothing to position signals"""
        smoothed_features = {}
        
        for key, signal in features.items():
            if key in ['is_left_fencer', 'frame_indices']:  # FIXED: Don't smooth frame indices
                smoothed_features[key] = signal
                continue
                
            if len(signal) >= self.smoothing_window:
                # Use Gaussian smoothing for better noise reduction
                smoothed_signal = gaussian_filter1d(signal, sigma=1.0)
                smoothed_features[key] = smoothed_signal
            else:
                smoothed_features[key] = signal
        
        return smoothed_features
    
    def compute_velocities(self, features):
        """Compute velocities and accelerations for movement analysis"""
        velocities = {}
        accelerations = {}
        
        for key, position in features.items():
            if key in ['is_left_fencer', 'frame_indices']:  # FIXED: Skip non-position data
                continue
            
            # Check if array has enough elements for gradient calculation
            # numpy.gradient requires at least 2 elements (edge_order + 1)
            if len(position) < 2:
                logging.warning(f"Array {key} has {len(position)} elements, too small for gradient calculation. Skipping.")
                velocities[key] = np.zeros_like(position)
                accelerations[key] = np.zeros_like(position)
                continue
                
            # Compute velocity (first derivative)
            velocity = np.gradient(position)
            velocities[key] = velocity
            
            # Compute acceleration (second derivative)
            acceleration = np.gradient(velocity)
            accelerations[key] = acceleration
        
        return velocities, accelerations
    
    def detect_movement_phases(self, features, velocities, accelerations):
        """
        Detect movement phases using multiple signals and statistical analysis
        
        Returns:
        - movement_labels: Array with labels 0=pause, 1=advance, -1=retreat
        """
        n_frames = len(features['weighted_position'])
        movement_labels = np.zeros(n_frames)
        
        # Primary signals for movement detection
        primary_velocity = velocities['weighted_position']
        front_foot_velocity = velocities['front_foot']
        back_foot_velocity = velocities['back_foot']
        hip_velocity = velocities['hip_center']
        
        # Movement direction indicator (positive = forward for the fencer)
        if features['is_left_fencer']:
            direction_multiplier = 1  # Left fencer moves right (positive x)
        else:
            direction_multiplier = -1  # Right fencer moves left (negative x)
        
        # Adjust velocities for fencer direction
        adj_primary_vel = primary_velocity * direction_multiplier
        adj_front_vel = front_foot_velocity * direction_multiplier
        adj_back_vel = back_foot_velocity * direction_multiplier
        adj_hip_vel = hip_velocity * direction_multiplier
        
        # Adaptive thresholding based on signal statistics
        vel_std = np.std(adj_primary_vel)
        adaptive_threshold = max(self.velocity_threshold, 0.3 * vel_std)
        
        for i in range(n_frames):
            # Multi-signal voting approach
            advance_votes = 0
            retreat_votes = 0
            pause_votes = 0
            
            # Vote 1: Primary weighted position
            if adj_primary_vel[i] > adaptive_threshold:
                advance_votes += 2  # Higher weight
            elif adj_primary_vel[i] < -adaptive_threshold:
                retreat_votes += 2
            else:
                pause_votes += 1
            
            # Vote 2: Front foot movement
            if adj_front_vel[i] > adaptive_threshold * 0.8:
                advance_votes += 1
            elif adj_front_vel[i] < -adaptive_threshold * 0.8:
                retreat_votes += 1
            else:
                pause_votes += 1
            
            # Vote 3: Hip movement (body commitment)
            if adj_hip_vel[i] > adaptive_threshold * 0.6:
                advance_votes += 1
            elif adj_hip_vel[i] < -adaptive_threshold * 0.6:
                retreat_votes += 1
            else:
                pause_votes += 1
            
            # Vote 4: Back foot movement (retreat indicator)
            if adj_back_vel[i] < -adaptive_threshold * 0.5:  # Back foot moving backward
                retreat_votes += 2
            elif adj_back_vel[i] > adaptive_threshold * 0.3:  # Back foot following forward
                advance_votes += 1
            
            # Determine movement based on votes
            if advance_votes > max(retreat_votes, pause_votes):
                movement_labels[i] = 1  # Advance
            elif retreat_votes > max(advance_votes, pause_votes):
                movement_labels[i] = -1  # Retreat
            else:
                movement_labels[i] = 0  # Pause
        
        return movement_labels
    
    def refine_intervals(self, movement_labels):
        """
        Refine movement intervals by:
        1. Removing very short intervals
        2. Merging similar adjacent intervals
        3. Applying temporal consistency
        """
        refined_labels = movement_labels.copy()
        n_frames = len(refined_labels)
        
        # Apply median filter to reduce noise
        window = min(5, n_frames // 10)
        if window >= 3:
            for i in range(window//2, n_frames - window//2):
                segment = refined_labels[i-window//2:i+window//2+1]
                refined_labels[i] = np.median(segment)
        
        # Merge very short intervals with neighbors
        intervals = self.extract_intervals_with_indices(refined_labels, None)  # Use row indices for internal processing
        
        for start_row, end_row, label in intervals:
            length = end_row - start_row + 1
            if length < self.min_interval_length:
                # Replace with most common neighbor label
                before_label = refined_labels[max(0, start_row-1)] if start_row > 0 else 0
                after_label = refined_labels[min(n_frames-1, end_row+1)] if end_row < n_frames-1 else 0
                
                # Choose the more common surrounding label
                if before_label == after_label:
                    refined_labels[start_row:end_row+1] = before_label
                else:
                    # Use the label that appears more in a larger window
                    window_start = max(0, start_row - 10)
                    window_end = min(n_frames, end_row + 10)
                    window_labels = np.concatenate([
                        refined_labels[window_start:start_row],
                        refined_labels[end_row+1:window_end]
                    ])
                    if len(window_labels) > 0:
                        most_common = np.bincount(window_labels.astype(int) + 1).argmax() - 1
                        refined_labels[start_row:end_row+1] = most_common
        
        return refined_labels
    
    def extract_intervals_with_indices(self, movement_labels, frame_indices):
        """
        FIXED: Extract intervals and convert to real frame indices
        
        Parameters:
        - movement_labels: Array of movement labels
        - frame_indices: Array of real frame numbers (or None for row indices)
        
        Returns:
        - List of (start_frame, end_frame, label) tuples using REAL frame indices
        """
        if len(movement_labels) == 0:
            return []
        
        intervals = []
        current_label = movement_labels[0]
        start_row = 0
        
        for i in range(1, len(movement_labels)):
            if movement_labels[i] != current_label:
                # Convert row indices to real frame indices
                if frame_indices is not None:
                    start_frame = int(frame_indices[start_row])
                    end_frame = int(frame_indices[i-1])
                else:
                    start_frame = start_row
                    end_frame = i-1
                    
                intervals.append((start_frame, end_frame, current_label))
                start_row = i
                current_label = movement_labels[i]
        
        # Add the last interval
        if frame_indices is not None:
            start_frame = int(frame_indices[start_row])
            end_frame = int(frame_indices[len(movement_labels)-1])
        else:
            start_frame = start_row
            end_frame = len(movement_labels)-1
            
        intervals.append((start_frame, end_frame, current_label))
        
        return intervals
    
    def extract_intervals(self, movement_labels):
        """Legacy method - kept for compatibility but now returns row indices"""
        return self.extract_intervals_with_indices(movement_labels, None)
    
    def categorize_intervals(self, refined_labels, frame_indices):
        """
        FIXED: Categorize intervals into advance, pause, and retreat using REAL frame indices
        
        Parameters:
        - refined_labels: Array of refined movement labels
        - frame_indices: Array of real frame numbers
        
        Returns:
        - advance_intervals: List of (start_frame, end_frame) tuples with REAL frame indices
        - pause_intervals: List of (start_frame, end_frame) tuples with REAL frame indices
        - retreat_intervals: List of (start_frame, end_frame) tuples with REAL frame indices
        """
        intervals = self.extract_intervals_with_indices(refined_labels, frame_indices)
        
        advance_intervals = []
        pause_intervals = []
        retreat_intervals = []
        
        for start_frame, end_frame, label in intervals:
            # Calculate length in terms of row count for min_interval_length check
            start_row = np.where(frame_indices == start_frame)[0][0]
            end_row = np.where(frame_indices == end_frame)[0][0]
            length = end_row - start_row + 1
            
            if length >= self.min_interval_length:
                if label == 1:
                    advance_intervals.append((start_frame, end_frame))
                elif label == 0:
                    pause_intervals.append((start_frame, end_frame))
                elif label == -1:
                    retreat_intervals.append((start_frame, end_frame))
        
        return advance_intervals, pause_intervals, retreat_intervals
    
    def analyze_movement_patterns(self, xdata_df, is_left_fencer=True, plot=False):
        """
        FIXED: Main function to analyze movement patterns with real frame indices
        
        Parameters:
        - xdata_df: DataFrame with keypoint x-coordinates
        - is_left_fencer: Boolean indicating fencer side
        - plot: Whether to create visualization plots
        
        Returns:
        - Dict with intervals using REAL frame indices and analysis results
        """
        # Extract features
        features = self.extract_position_features(xdata_df, is_left_fencer)
        if features is None:
            return None
        
        # Check if we have enough data points for analysis
        # Need at least 3 frames for meaningful movement analysis
        n_frames = len(features.get('weighted_position', []))
        if n_frames < 3:
            logging.warning(f"Insufficient data for movement analysis: only {n_frames} frames. Minimum 3 required.")
            return {
                'advance_intervals': [],
                'pause_intervals': [],
                'retreat_intervals': [],
                'movement_labels': np.zeros(n_frames),
                'frame_indices': features.get('frame_indices', np.arange(n_frames)),
                'features': features,
                'velocities': {},
                'total_frames': n_frames,
                'is_left_fencer': is_left_fencer
            }
        
        # Smooth signals
        smoothed_features = self.smooth_signals(features)
        
        # Compute velocities and accelerations
        velocities, accelerations = self.compute_velocities(smoothed_features)
        
        # Detect movement phases
        movement_labels = self.detect_movement_phases(smoothed_features, velocities, accelerations)
        
        # Refine intervals
        refined_labels = self.refine_intervals(movement_labels)
        
        # FIXED: Categorize intervals using real frame indices
        advance_intervals, pause_intervals, retreat_intervals = self.categorize_intervals(
            refined_labels, smoothed_features['frame_indices']
        )
        
        # Create result dictionary
        result = {
            'advance_intervals': advance_intervals,        # FIXED: Now contains real frame indices
            'pause_intervals': pause_intervals,            # FIXED: Now contains real frame indices
            'retreat_intervals': retreat_intervals,        # FIXED: Now contains real frame indices
            'movement_labels': refined_labels,
            'frame_indices': smoothed_features['frame_indices'],  # FIXED: Include frame mapping
            'features': smoothed_features,
            'velocities': velocities,
            'total_frames': len(xdata_df),
            'is_left_fencer': is_left_fencer
        }
        
        # Optional plotting
        if plot:
            self.plot_movement_analysis(result, xdata_df)
        
        return result
    
    def plot_movement_analysis(self, result, xdata_df):
        """FIXED: Create visualization of movement analysis with real frame indices"""
        fig, axes = plt.subplots(4, 1, figsize=(15, 12))
        
        # FIXED: Use real frame indices for x-axis
        frames = result['frame_indices']
        features = result['features']
        velocities = result['velocities']
        labels = result['movement_labels']
        
        # Plot 1: Position signals
        axes[0].plot(frames, features['weighted_position'], label='Weighted Position', linewidth=2)
        axes[0].plot(frames, features['front_foot'], label='Front Foot', alpha=0.7)
        axes[0].plot(frames, features['back_foot'], label='Back Foot', alpha=0.7)
        axes[0].plot(frames, features['hip_center'], label='Hip Center', alpha=0.7)
        axes[0].set_ylabel('X Position')
        axes[0].set_title('Position Signals')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # Plot 2: Velocity signals
        direction_mult = 1 if result['is_left_fencer'] else -1
        axes[1].plot(frames, velocities['weighted_position'] * direction_mult, 
                    label='Weighted Velocity', linewidth=2)
        axes[1].plot(frames, velocities['front_foot'] * direction_mult, 
                    label='Front Foot Velocity', alpha=0.7)
        axes[1].plot(frames, velocities['back_foot'] * direction_mult, 
                    label='Back Foot Velocity', alpha=0.7)
        axes[1].axhline(y=self.velocity_threshold, color='red', linestyle='--', 
                       label='Threshold', alpha=0.5)
        axes[1].axhline(y=-self.velocity_threshold, color='red', linestyle='--', alpha=0.5)
        axes[1].set_ylabel('Velocity')
        axes[1].set_title('Velocity Signals (Adjusted for Fencer Direction)')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        
        # Plot 3: Movement labels
        axes[2].plot(frames, labels, linewidth=3)
        axes[2].set_ylabel('Movement Type')
        axes[2].set_title('Detected Movement Patterns')
        axes[2].set_yticks([-1, 0, 1])
        axes[2].set_yticklabels(['Retreat', 'Pause', 'Advance'])
        axes[2].grid(True, alpha=0.3)
        
        # Plot 4: Interval visualization with REAL frame indices
        y_pos = 0
        colors = {'advance': 'green', 'pause': 'orange', 'retreat': 'red'}
        
        for interval_type, intervals in [('advance', result['advance_intervals']),
                                       ('pause', result['pause_intervals']),
                                       ('retreat', result['retreat_intervals'])]:
            for start, end in intervals:
                # FIXED: Now start and end are already real frame indices
                axes[3].barh(y_pos, end - start + 1, left=start, height=0.8,
                           color=colors[interval_type], alpha=0.7, 
                           label=interval_type if start == intervals[0][0] else "")
            y_pos += 1
        
        axes[3].set_xlabel('Frame')
        axes[3].set_ylabel('Interval Type')
        axes[3].set_title('Movement Intervals (Real Frame Indices)')
        axes[3].set_yticks([0, 1, 2])
        axes[3].set_yticklabels(['Advance', 'Pause', 'Retreat'])
        axes[3].legend()
        axes[3].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
        
        # FIXED: Print summary with real frame indices
        print(f"\nMovement Analysis Summary:")
        print(f"Total frames: {result['total_frames']}")
        print(f"Advance intervals: {len(result['advance_intervals'])}")
        print(f"Pause intervals: {len(result['pause_intervals'])}")
        print(f"Retreat intervals: {len(result['retreat_intervals'])}")
        
        # Calculate durations using real frame indices
        total_advance = sum(end - start + 1 for start, end in result['advance_intervals'])
        total_pause = sum(end - start + 1 for start, end in result['pause_intervals'])
        total_retreat = sum(end - start + 1 for start, end in result['retreat_intervals'])
        
        print(f"Time advancing: {total_advance} frames ({100*total_advance/result['total_frames']:.1f}%)")
        print(f"Time pausing: {total_pause} frames ({100*total_pause/result['total_frames']:.1f}%)")
        print(f"Time retreating: {total_retreat} frames ({100*total_retreat/result['total_frames']:.1f}%)")
        
        # FIXED: Print actual frame ranges
        print(f"\nAdvance intervals (real frames):")
        for i, (start, end) in enumerate(result['advance_intervals']):
            print(f"  Advance {i+1}: frames {start}-{end} (duration: {end-start+1})")
        
        print(f"\nRetreat intervals (real frames):")
        for i, (start, end) in enumerate(result['retreat_intervals']):
            print(f"  Retreat {i+1}: frames {start}-{end} (duration: {end-start+1})")

# FIXED: Easy-to-use function with real frame indices
def detect_fencing_movements(xdata_df, is_left_fencer=True, velocity_threshold=0.03, 
                           min_interval_length=5, plot=True):
    """
    FIXED: Detect fencing movement patterns from keypoint data with real frame indices
    
    Parameters:
    - xdata_df: DataFrame with keypoint x-coordinates (must have 'Frame' column for real indices)
    - is_left_fencer: True for left fencer, False for right fencer
    - velocity_threshold: Sensitivity for movement detection
    - min_interval_length: Minimum frames for valid intervals
    - plot: Whether to show analysis plots
    
    Returns:
    - Dictionary with advance_intervals, pause_intervals, retreat_intervals using REAL frame indices
    """
    detector = FencingMovementDetector(
        velocity_threshold=velocity_threshold,
        min_interval_length=min_interval_length
    )
    
    return detector.analyze_movement_patterns(xdata_df, is_left_fencer, plot=plot)

# Usage example:
"""
# Now returns real frame indices instead of row indices
result = detect_fencing_movements(xdata_df, is_left_fencer=True, plot=True)

# Advance intervals now contain actual frame numbers
for start_frame, end_frame in result['advance_intervals']:
    print(f"Advance from frame {start_frame} to {end_frame}")
    
# These frame numbers can be used directly for video analysis or other processing
"""

def convert_numpy_types(obj):
    """Convert numpy types to native Python types for JSON serialization."""
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, np.bool_):
        return bool(obj)
    elif isinstance(obj, dict):
        return {key: convert_numpy_types(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_types(item) for item in obj]
    elif isinstance(obj, tuple):
        return tuple(convert_numpy_types(item) for item in obj)
    else:
        return obj

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s',
    handlers=[
        logging.FileHandler('/workspace/Project/bout_analysis.log'),
        logging.StreamHandler()
    ]
)

# Use REST helper elsewhere to avoid gRPC DNS issues

def find_stop_and_back_intervals(xdata_df, is_left_fencer, flat_threshold=0.03, min_length=5, window_size=5):
    """Identify advance and pause/retreat intervals based on fencer movement."""
    x11 = xdata_df['11'].values
    x12 = xdata_df['12'].values
    x16 = xdata_df['16'].values
    x15 = xdata_df['15'].values
    p_hips = (x11 + x12) / 2
    p_weighted = 0.7 * x16 + 0.3 * p_hips
    if len(p_weighted) >= window_size:
        p_smooth = savgol_filter(p_weighted, window_size, 3)
    else:
        p_smooth = p_weighted
    if len(x15) >= window_size:
        back_foot_smooth = savgol_filter(x15, window_size, 3)
    else:
        back_foot_smooth = x15
    v_weighted = np.diff(p_smooth)
    v_weighted = np.insert(v_weighted, 0, 0)
    v_back_foot = np.diff(back_foot_smooth)
    v_back_foot = np.insert(v_back_foot, 0, 0)
    intervals = []
    start = 0
    current_v_weighted = [v_weighted[0]]
    current_v_back = [v_back_foot[0]]
    for t in range(1, len(v_weighted)):
        is_pause = abs(v_weighted[t]) < flat_threshold
        if is_left_fencer:
            is_back = v_back_foot[t] < 0
        else:
            is_back = v_back_foot[t] > 0
        frame_category = 'pause/retreat' if is_pause or is_back else 'advance'
        avg_v_weighted = np.mean(current_v_weighted)
        avg_v_back = np.mean(current_v_back)
        is_interval_pause = abs(avg_v_weighted) < flat_threshold
        if is_left_fencer:
            is_interval_back = avg_v_back < 0
        else:
            is_interval_back = avg_v_back > 0
        interval_category = 'pause/retreat' if is_interval_pause or is_interval_back else 'advance'
        if frame_category != interval_category:
            if start > 5:
                intervals.append((start, t-1, interval_category))
            start = t
            current_v_weighted = [v_weighted[t]]
            current_v_back = [v_back_foot[t]]
        else:
            current_v_weighted.append(v_weighted[t])
            current_v_back.append(v_back_foot[t])
    if current_v_weighted and start > 5:
        avg_v_weighted = np.mean(current_v_weighted)
        avg_v_back = np.mean(current_v_back)
        is_interval_pause = abs(avg_v_weighted) < flat_threshold
        if is_left_fencer:
            is_interval_back = avg_v_back < 0
        else:
            is_interval_back = avg_v_back > 0
        interval_category = 'pause/retreat' if is_interval_pause or is_interval_back else 'advance'
        intervals.append((start, len(v_weighted)-1, interval_category))
    merged_intervals = []
    if not intervals:
        return [], []
    current_start, current_end, current_category = intervals[0]
    for start, end, category in intervals[1:]:
        if category == current_category:
            current_end = end
        else:
            merged_intervals.append((current_start, current_end, current_category))
            current_start, current_end, current_category = start, end, category
    merged_intervals.append((current_start, current_end, current_category))
    advance_intervals = []
    pause_retreat_intervals = []
    for start, end, category in merged_intervals:
        if (end - start + 1) >= min_length:
            if category == 'advance':
                advance_intervals.append((start, end))
            elif category == 'pause/retreat':
                pause_retreat_intervals.append((start, end))
    return advance_intervals, pause_retreat_intervals

def compute_first_step_metrics(left_x_df, right_x_df, fps=30, velocity_threshold=0.01, window_size=5):
    """Compute first step initiation time, velocity, acceleration, and momentum for both fencers."""
    logging.debug("Computing first step metrics")
    left_x = left_x_df['16'].values
    right_x = right_x_df['16'].values
    if len(left_x) >= window_size:
        left_x_smooth = savgol_filter(left_x, window_size, 2)
        right_x_smooth = savgol_filter(right_x, window_size, 2)
    else:
        left_x_smooth = left_x
        right_x_smooth = right_x
    left_v = np.diff(left_x_smooth)
    left_v = np.insert(left_v, 0, 0)
    right_v = np.diff(right_x_smooth)
    right_v = np.insert(right_v, 0, 0)
    left_first_step = None
    right_first_step = None
    for i in range(len(left_v)):
        if left_v[i] > velocity_threshold:
            left_first_step = i
            break
    for i in range(len(right_v)):
        if right_v[i] < -velocity_threshold:
            right_first_step = i
            break
    left_metrics = {'init_time': None, 'velocity': 0.0, 'acceleration': 0.0, 'momentum': 0.0}
    right_metrics = {'init_time': None, 'velocity': 0.0, 'acceleration': 0.0, 'momentum': 0.0}
    if left_first_step is not None:
        left_metrics['init_time'] = left_first_step / fps
        end_idx = min(left_first_step + 8, len(left_x_smooth) - 1)
        if end_idx > left_first_step:
            v = left_x_smooth[left_first_step:end_idx] - left_x_smooth[left_first_step-1:end_idx-1]
            left_metrics['velocity'] = np.mean(v) * fps
            a = np.diff(v) * fps * fps
            left_metrics['acceleration'] = np.mean(a) if len(a) > 0 else 0.0
            left_metrics['momentum'] = abs(left_metrics['velocity'])  # Approximate momentum as velocity
    if right_first_step is not None:
        right_metrics['init_time'] = right_first_step / fps
        end_idx = min(right_first_step + 8, len(right_x_smooth) - 1)
        if end_idx > right_first_step:
            v = right_x_smooth[right_first_step:end_idx] - right_x_smooth[right_first_step-1:end_idx-1]
            right_metrics['velocity'] = -np.mean(v) * fps
            a = np.diff(v) * fps * fps
            right_metrics['acceleration'] = -np.mean(a) if len(a) > 0 else 0.0
            right_metrics['momentum'] = abs(right_metrics['velocity'])
    logging.debug(f"Left first step metrics: {left_metrics}")
    logging.debug(f"Right first step metrics: {right_metrics}")
    return left_metrics, right_metrics

def calculate_velocity_and_acceleration(left_xdata_df, right_xdata_df, video_angle, left_has_launch=False, right_has_launch=False, frame_range=None, window_size=5, scale_factor=0.75, fps=30):
    """Calculate velocity and acceleration for the active attack phase."""
    def compute_metrics(xdata_df, has_launch, is_left_fencer):
        if xdata_df.empty:
            return 0.0, 0.0
        positions = xdata_df['16'].values
        frames = xdata_df['Frame'].values
        if len(positions) >= window_size:
            p_smooth = savgol_filter(positions, window_size, 2)
        else:
            p_smooth = positions
        v = np.diff(p_smooth)
        v = np.insert(v, 0, 0)
        if has_launch:
            start_frame, end_frame = frame_range
            launch_start = max(start_frame, end_frame - 15 + 1)
            launch_indices = (frames >= launch_start) & (frames <= end_frame)
            if not np.any(launch_indices):
                return 0.0, 0.0
            launch_positions = p_smooth[launch_indices]
            launch_frames = frames[launch_indices]
            if is_left_fencer:
                peak_idx = np.argmax(launch_positions)
            else:
                peak_idx = np.argmin(launch_positions)
            active_end_frame = launch_frames[peak_idx]
            active_indices = (frames >= start_frame) & (frames <= active_end_frame)
        else:
            active_indices = np.ones(len(frames), dtype=bool)
        active_positions = p_smooth[active_indices]
        active_frames = frames[active_indices]
        if len(active_positions) < 2:
            return 0.0, 0.0
        active_v = np.diff(active_positions)
        active_v = np.insert(active_v, 0, 0)
        time_diffs = np.diff(active_frames) / fps
        time_diffs = np.insert(time_diffs, 0, 1/fps)
        avg_velocity = np.sum(active_v) / (active_frames[-1] - active_frames[0]) * fps if active_frames[-1] != active_frames[0] else 0.0
        a = np.diff(active_v) / time_diffs[1:]
        a = np.insert(a, 0, 0)
        avg_acceleration = np.mean(a) if len(a) > 0 else 0.0
        angle_aligned = (is_left_fencer and video_angle == 'left') or (not is_left_fencer and video_angle == 'right')
        if angle_aligned:
            avg_velocity *= scale_factor
            avg_acceleration *= scale_factor
        return abs(avg_velocity), abs(avg_acceleration)
    if not left_xdata_df.empty and not right_xdata_df.empty:
        left_x16 = left_xdata_df['16'].values
        right_x16 = right_xdata_df['16'].values
        frames = left_xdata_df['Frame'].values
        if len(left_x16) >= window_size:
            left_x16 = savgol_filter(left_x16, window_size, 2)
            right_x16 = savgol_filter(right_x16, window_size, 2)
        distances = np.abs(left_x16 - right_x16)
        meeting_idx = np.argmin(distances)
        meeting_frame = frames[meeting_idx]
        left_indices = left_xdata_df['Frame'] <= meeting_frame
        right_indices = right_xdata_df['Frame'] <= meeting_frame
        if not left_has_launch:
            left_xdata_df = left_xdata_df[left_indices]
        if not right_has_launch:
            right_xdata_df = right_xdata_df[right_indices]
    else:
        meeting_frame = frame_range[1] if frame_range else 0
    left_avg_velocity, left_avg_acceleration = compute_metrics(left_xdata_df, left_has_launch, True)
    right_avg_velocity, right_avg_acceleration = compute_metrics(right_xdata_df, right_has_launch, False)
    return (left_avg_velocity, left_avg_acceleration, right_avg_velocity, right_avg_acceleration)

def calculate_arm_extensions(left_xdata_new_df, left_ydata_new_df, right_xdata_new_df, right_ydata_new_df, match_data_dir, match_idx):
    """Identify arm extension intervals for potential attacks."""
    if left_xdata_new_df.empty or right_xdata_new_df.empty:
        logging.warning("Empty DataFrame in calculate_arm_extensions")
        return [], []
    left_x_df = left_xdata_new_df.copy()
    left_y_df = left_ydata_new_df.copy()
    right_x_df = right_xdata_new_df.copy()
    right_y_df = right_ydata_new_df.copy()
    corrections = []
    def correct_arm_keypoints(df, col, fencer, coord, corrections):
        original_values = df[col].values
        non_zero_values = original_values[original_values > 0]
        threshold = np.percentile(non_zero_values, 10) if len(non_zero_values) > 0 else 0
        logging.debug(f"{fencer} {coord} keypoint {col}: Anomaly threshold = {threshold:.6f}")
        anomaly_mask = (original_values == 0) | (original_values < threshold)
        if np.any(anomaly_mask):
            anomaly_frames = np.where(anomaly_mask)[0]
            logging.debug(f"Detected anomalous values in {fencer} {coord} keypoint {col} at frames: {list(anomaly_frames)}")
            df.loc[anomaly_mask, col] = np.nan
            df[col] = df[col].interpolate(method='linear', limit_direction='both')
            for frame in anomaly_frames:
                original = float(original_values[frame])
                corrected = float(df.loc[frame, col])
                corrections.append({
                    'fencer': fencer,
                    'coord': coord,
                    'column': col,
                    'frame': int(frame),
                    'original': original,
                    'corrected': corrected
                })
                logging.debug(f"Corrected {fencer} {coord} keypoint {col} at frame {frame}: {original:.6f} -> {corrected:.6f}")
        df[col] = df[col].fillna(method='ffill').fillna(method='bfill')
        return df
    left_x_df = correct_arm_keypoints(left_x_df, '10', 'Left', 'x', corrections)
    left_y_df = correct_arm_keypoints(left_y_df, '10', 'Left', 'y', corrections)
    right_x_df = correct_arm_keypoints(right_x_df, '10', 'Right', 'x', corrections)
    right_y_df = correct_arm_keypoints(right_y_df, '10', 'Right', 'y', corrections)
    left_x_df = correct_arm_keypoints(left_x_df, '12', 'Left', 'x', corrections)
    left_y_df = correct_arm_keypoints(left_y_df, '12', 'Left', 'y', corrections)
    right_x_df = correct_arm_keypoints(right_x_df, '12', 'Right', 'x', corrections)
    right_y_df = correct_arm_keypoints(right_y_df, '12', 'Right', 'y', corrections)
    left_hand_x = left_x_df['10'].values
    left_hip_x = left_x_df['12'].values
    left_hand_body_distance = np.abs(left_hand_x - left_hip_x)
    right_hand_x = right_x_df['10'].values
    right_hip_x = right_x_df['12'].values
    right_hand_body_distance = np.abs(right_hand_x - right_hip_x)
    if np.any(np.isnan(left_hand_body_distance)) or np.any(np.isnan(right_hand_body_distance)):
        logging.warning("NaN values in hand-body distance after correction")
        return [], []
    left_threshold = np.mean(left_hand_body_distance) * 1.3
    right_threshold = np.mean(right_hand_body_distance) * 1.3
    def find_extension_intervals(distances, threshold, df):
        intervals = []
        start = None
        for i in range(len(distances)):
            try:
                if distances[i] > threshold:
                    if start is None:
                        start = int(i)
                else:
                    if start is not None:
                        end = int(i - 1)
                        if end >= start and start < len(df) and end < len(df):
                            frame_start = int(df.iloc[start]['Frame'])
                            frame_end = int(df.iloc[end]['Frame'])
                            if frame_start <= frame_end:
                                intervals.append((frame_start, frame_end))
                        start = None
            except Exception as e:
                logging.error(f"Error processing distance at index {i}: {str(e)}\n{traceback.format_exc()}")
                continue
        if start is not None:
            end = int(len(distances) - 1)
            if end >= start and start < len(df) and end < len(df):
                frame_start = int(df.iloc[start]['Frame'])
                frame_end = int(df.iloc[end]['Frame'])
                if frame_start <= frame_end:
                    intervals.append((frame_start, frame_end))
        logging.debug(f"Extension intervals: {intervals}")
        return intervals
    left_arm_extensions = find_extension_intervals(left_hand_body_distance, left_threshold, left_x_df)
    right_arm_extensions = find_extension_intervals(right_hand_body_distance, right_threshold, right_x_df)
    corrections_path = os.path.join(match_data_dir, f"match_{match_idx}", 'arm_corrections.json')
    os.makedirs(os.path.dirname(corrections_path), exist_ok=True)
    with open(corrections_path, 'w', encoding='utf-8') as f:
        json.dump(corrections, f, indent=4)
    logging.info(f"Saved arm corrections to {corrections_path}")
    return left_arm_extensions, right_arm_extensions

def convert_intervals_to_seconds(intervals, fps=30):
    """Convert frame intervals to seconds."""
    return [(round(start / fps, 2), round(end / fps, 2)) for start, end in intervals]

def calculate_overall_velocity_acceleration(xdata_df, is_left_fencer=True, video_angle='unknown', window_size=5, scale_factor=0.75, fps=30):
    """
    Calculate overall velocity and acceleration for the entire bout data.
    
    Parameters:
    - xdata_df: DataFrame with keypoint x-coordinates and Frame column
    - is_left_fencer: Boolean indicating fencer side
    - video_angle: Video angle for scaling adjustments
    - window_size: Window size for smoothing
    - scale_factor: Scaling factor for angle alignment
    - fps: Frames per second
    
    Returns:
    - Tuple of (overall_velocity, overall_acceleration)
    """
    if xdata_df.empty or len(xdata_df) < 2:
        return 0.0, 0.0
    
    # Use front foot position (keypoint 16) for overall movement analysis
    positions = xdata_df['16'].values
    frames = xdata_df['Frame'].values
    
    # Apply smoothing if we have enough data points
    if len(positions) >= window_size:
        positions_smooth = savgol_filter(positions, window_size, 2)
    else:
        positions_smooth = positions
    
    # Calculate velocity (first derivative)
    velocities = np.diff(positions_smooth)
    velocities = np.insert(velocities, 0, 0)  # Add initial zero velocity
    
    # Calculate time differences between frames
    time_diffs = np.diff(frames) / fps
    time_diffs = np.insert(time_diffs, 0, 1/fps)  # Add initial time step
    
    # Calculate acceleration (second derivative)
    accelerations = np.diff(velocities) / time_diffs[1:]
    accelerations = np.insert(accelerations, 0, 0)  # Add initial zero acceleration
    
    # Calculate overall average velocity and acceleration
    total_displacement = positions_smooth[-1] - positions_smooth[0]
    total_time = (frames[-1] - frames[0]) / fps
    overall_velocity = total_displacement / total_time if total_time > 0 else 0.0
    
    # Average acceleration over the entire bout
    overall_acceleration = np.mean(np.abs(accelerations)) if len(accelerations) > 0 else 0.0
    
    # Apply video angle scaling
    angle_aligned = (is_left_fencer and video_angle == 'left') or (not is_left_fencer and video_angle == 'right')
    if angle_aligned:
        overall_velocity *= scale_factor
        overall_acceleration *= scale_factor
    
    # For right fencer, consider direction (moving left is negative)
    if not is_left_fencer:
        overall_velocity *= -1
    
    return abs(overall_velocity), overall_acceleration

def detect_launch(xdata_df, ydata_df, is_left_fencer, frame_range, video_angle, last_n_frames=15, distance_multiplier=1.4):
    """Detect if a fencer launches in the last 15 frames."""
    if xdata_df.empty:
        logging.warning("xdata_df is empty, cannot detect launch")
        return False, -1
    start_frame, end_frame = frame_range
    launch_start = max(start_frame, end_frame - last_n_frames + 1)
    bout_df = xdata_df[(xdata_df['Frame'] >= start_frame) & (xdata_df['Frame'] <= end_frame)]
    launch_df = xdata_df[(xdata_df['Frame'] >= launch_start) & (xdata_df['Frame'] <= end_frame)]
    if launch_df.empty:
        logging.debug(f"No data in the last {last_n_frames} frames")
        return False, -1
    angle_aligned = (is_left_fencer and video_angle == 'left') or (not is_left_fencer and video_angle == 'right')
    adjusted_multiplier = distance_multiplier * 0.8 if angle_aligned else distance_multiplier
    front_foot_x = launch_df['16'].values
    back_foot_x = launch_df['15'].values
    foot_distance = np.abs(front_foot_x - back_foot_x)
    historical_df = bout_df[bout_df['Frame'] < end_frame - 15]
    if not historical_df.empty:
        historical_front_x = historical_df['16'].values
        historical_back_x = historical_df['15'].values
        historical_distance = np.abs(historical_front_x - historical_back_x)
        avg_historical_distance = np.mean(historical_distance) if len(historical_distance) > 0 else 0
    else:
        avg_historical_distance = 0
    distance_threshold = avg_historical_distance * adjusted_multiplier if avg_historical_distance > 0 else None
    logging.debug(f"Launch detection threshold: {distance_threshold}")
    for i in range(len(foot_distance)):
        distance_condition = foot_distance[i] > distance_threshold if avg_historical_distance > 0 else False
        if distance_condition:
            launch_frame = int(launch_df['Frame'].iloc[i])
            logging.debug(f"Launch detected at frame {launch_frame}")
            return True, launch_frame
    logging.debug("No launch detected")
    return False, -1

def calculate_velocity_and_acceleration_interval(xdata_df, is_left_fencer, start_frame, end_frame, video_angle, window_size=5, scale_factor=0.9, fps=30):
    """Calculate velocity and acceleration for a specific interval."""
    if xdata_df.empty:
        logging.warning("DataFrame is empty")
        return 0.0, 0.0
    interval_df = xdata_df[(xdata_df['Frame'] >= start_frame) & (xdata_df['Frame'] <= end_frame)]
    if interval_df.empty:
        logging.debug(f"No data in frame range {start_frame} to {end_frame}")
        return 0.0, 0.0
    x16 = interval_df['16'].values
    frames = interval_df['Frame'].values
    if len(x16) >= window_size:
        p_smooth = savgol_filter(x16, window_size, 2)
    else:
        p_smooth = x16
    v = np.diff(p_smooth)
    v = np.insert(v, 0, 0)
    active_positions = p_smooth
    active_frames = frames
    if len(active_positions) < 2:
        return 0.0, 0.0
    active_v = np.diff(active_positions)
    active_v = np.insert(active_v, 0, 0)
    time_diffs = np.diff(active_frames) / fps
    time_diffs = np.insert(time_diffs, 0, 1/fps)
    avg_velocity = np.sum(active_v) / (active_frames[-1] - active_frames[0]) * fps if active_frames[-1] != active_frames[0] else 0.0
    a = np.diff(active_v) / time_diffs[1:]
    a = np.insert(a, 0, 0)
    avg_acceleration = np.mean(a) if len(a) > 0 else 0.0
    if video_angle == 'right' and not is_left_fencer:
        avg_velocity *= scale_factor
        avg_acceleration *= scale_factor
    elif video_angle == 'left' and is_left_fencer:
        avg_velocity *= scale_factor
        avg_acceleration *= scale_factor
    if not is_left_fencer:
        avg_acceleration *= -1
        avg_velocity *= -1
    return avg_velocity, avg_acceleration

import numpy as np
import pandas as pd
from scipy.signal import savgol_filter
import math

class FencingIntervalAnalyzer:
    """Advanced analysis for fencing advance and retreat intervals"""
    
    def __init__(self, fps=30, distance_threshold=1.5, tempo_variation_threshold=0.3, min_pause_frames=10):
        """
        Initialize analyzer with configurable parameters
        
        Parameters:
        - fps: Frames per second
        - distance_threshold: Good attacking distance in meters (default 2.0)
        - tempo_variation_threshold: Threshold for detecting tempo changes
        - min_pause_frames: Minimum consecutive frames of low velocity to be considered a pause
        """
        self.fps = fps
        self.distance_threshold = distance_threshold
        self.tempo_variation_threshold = tempo_variation_threshold
        self.min_pause_frames = min_pause_frames
    
    def calculate_fencer_distance(self, left_xdata, right_xdata, left_ydata, right_ydata, frame_indices):
        """
        Calculate distance between fencers for given frames
        
        Returns array of distances in meters (normalized by c factor)
        """
        # Use front foot positions (keypoint 16) for distance
        left_x16 = left_xdata['16'].values if isinstance(left_xdata, pd.DataFrame) else left_xdata
        right_x16 = right_xdata['16'].values if isinstance(right_xdata, pd.DataFrame) else right_xdata
        left_y16 = left_ydata['16'].values if isinstance(left_ydata, pd.DataFrame) else left_ydata
        right_y16 = right_ydata['16'].values if isinstance(right_ydata, pd.DataFrame) else right_ydata
        
        # Calculate Euclidean distance
        distances = np.sqrt((left_x16 - right_x16)**2 + (left_y16 - right_y16)**2)
        
        return distances
    
    def analyze_tempo_changes(self, xdata_df, interval_start, interval_end):
        """
        Analyze tempo changes within an interval by examining velocity variations.
        A "pause" is now defined as a period of low velocity lasting for a minimum
        number of consecutive frames.
        
        Returns:
        - tempo_changes: Number of significant tempo changes
        - has_pauses: Boolean indicating if there are micro-pauses
        - speed_variation: Coefficient of variation for speed
        """
        # Filter data for interval
        mask = (xdata_df['Frame'] >= interval_start) & (xdata_df['Frame'] <= interval_end)
        interval_data = xdata_df[mask]
        
        if len(interval_data) < self.min_pause_frames:
            return 0, False, 0.0
        
        # Calculate velocity from front foot movement
        front_foot = interval_data['16'].values
        velocity = np.abs(np.gradient(front_foot))
        
        # Smooth to reduce noise
        if len(velocity) >= 5:
            velocity = savgol_filter(velocity, min(5, len(velocity)), 2)
        
        # Detect tempo changes by finding significant velocity variations
        mean_velocity = np.mean(velocity)
        std_velocity = np.std(velocity)
        speed_variation = std_velocity / mean_velocity if mean_velocity > 0 else 0
        
        # Count significant changes
        tempo_changes = 0
        for i in range(1, len(velocity)-1):
            if abs(velocity[i] - velocity[i-1]) > self.tempo_variation_threshold * mean_velocity:
                tempo_changes += 1
        
        # NEW: Check for consecutive micro-pauses
        has_pauses = False
        pause_threshold = 0.1 * mean_velocity
        consecutive_pause_count = 0
        for v in velocity:
            if v < pause_threshold:
                consecutive_pause_count += 1
            else:
                consecutive_pause_count = 0  # Reset counter if velocity is above threshold
            
            if consecutive_pause_count >= self.min_pause_frames:
                has_pauses = True
                break  # A valid pause has been found, no need to check further
        
        return tempo_changes, has_pauses, speed_variation
    
    def classify_attack_type(self, arm_extensions, launches, advance_start, advance_end):
        """
        Classify the type of attack based on arm extensions and launches
        
        Returns attack type and characteristics
        """
        # Filter extensions and launches within the advance interval
        interval_extensions = []
        interval_launches = []
        
        for ext in arm_extensions:
            if advance_start <= ext['start_frame'] <= advance_end:
                interval_extensions.append(ext)
        
        for launch in launches:
            if advance_start <= launch['start_frame'] <= advance_end:
                interval_launches.append(launch)
        
        # Determine attack type
        attack_info = {
            'has_attack': False,
            'attack_type': 'none',
            'num_extensions': len(interval_extensions),
            'num_launches': len(interval_launches),
            'characteristics': []
        }
        
        if not interval_extensions and not interval_launches:
            attack_info['attack_type'] = 'no_attack'
            attack_info['characteristics'].append('No offensive action detected')
        elif interval_launches:
            attack_info['has_attack'] = True
            
            # Check if there are arm extensions near launches
            launch_with_extension = False
            for launch in interval_launches:
                for ext in interval_extensions:
                    # Check if extension overlaps with or is near launch (within 5 frames)
                    if ext['start_frame'] - launch['start_frame'] >= 0 and ext['start_frame'] - launch['start_frame'] <= 15:
                        launch_with_extension = True
                        break
            
            if launch_with_extension:
                if len(interval_extensions) > 1:
                    attack_info['attack_type'] = 'compound_attack'
                    attack_info['characteristics'].append('Multiple preparations before launch')
                else:
                    attack_info['attack_type'] = 'simple_attack'
                    attack_info['characteristics'].append('Direct attack with launch')
            else:
                attack_info['attack_type'] = 'holding_attack'
                attack_info['characteristics'].append('Launch without clear arm extension')
        elif interval_extensions:
            attack_info['has_attack'] = True
            if len(interval_extensions) > 1:
                attack_info['attack_type'] = 'compound_preparation'
                attack_info['characteristics'].append('Multiple feints/preparations without completion')
            else:
                attack_info['attack_type'] = 'simple_preparation'
                attack_info['characteristics'].append('Single preparation without completion')
        
        return attack_info

    
    def analyze_advance_interval(self, advance_interval, left_xdata_df, left_ydata_df, 
                               right_xdata_df, right_ydata_df, arm_extensions, launches,
                               is_left_attacking):
        """
        Comprehensive analysis of an advance interval
        
        Parameters:
        - advance_interval: Tuple (start_frame, end_frame)
        - Data for both fencers
        - arm_extensions: List of arm extension events
        - launches: List of launch events
        - is_left_attacking: Boolean indicating which fencer is advancing
        
        Returns detailed analysis dictionary
        """
        start_frame, end_frame = advance_interval
        
        # Calculate distances throughout the interval
        mask = (left_xdata_df['Frame'] >= start_frame) & (left_xdata_df['Frame'] <= end_frame)
        interval_left_x = left_xdata_df[mask]
        interval_left_y = left_ydata_df[mask]
        interval_right_x = right_xdata_df[mask]
        interval_right_y = right_ydata_df[mask]
        
        distances = self.calculate_fencer_distance(
            interval_left_x, interval_right_x, 
            interval_left_y, interval_right_y,
            interval_left_x['Frame'].values
        )
        
        # Analyze attack type
        if is_left_attacking:
            attack_info = self.classify_attack_type(
                arm_extensions, launches, start_frame, end_frame
            )
            tempo_changes, has_pauses, speed_var = self.analyze_tempo_changes(
                left_xdata_df, start_frame, end_frame
            )
        else:
            attack_info = self.classify_attack_type(
                arm_extensions, launches, start_frame, end_frame
            )
            tempo_changes, has_pauses, speed_var = self.analyze_tempo_changes(
                right_xdata_df, start_frame, end_frame
            )
        
        # Distance analysis (use None when unavailable to avoid non-JSON values)
        min_distance = np.min(distances) if len(distances) > 0 else None
        avg_distance = np.mean(distances) if len(distances) > 0 else None
        
        # Check for dangerous close distances without attack
        dangerous_frames = []
        for i, (d, frame) in enumerate(zip(distances, interval_left_x['Frame'].values)):
            if d < self.distance_threshold:
                # Check if there's an attack action at this frame
                has_action = False
                for ext in arm_extensions:
                    if ext['start_frame'] <= frame <= ext['end_frame']:
                        has_action = True
                        break
                for launch in launches:
                    if launch['start_frame'] <= frame <= launch['end_frame']:
                        has_action = True
                        break
                
                if not has_action:
                    dangerous_frames.append(frame)
        
        # Tempo classification
        if tempo_changes > 15 or has_pauses:
            tempo_type = 'broken_tempo'
            tempo_desc = 'Changing rhythm with pauses/accelerations'
        elif speed_var > 0.5:
            tempo_type = 'variable_tempo'
            tempo_desc = 'Varying speed throughout advance'
        else:
            tempo_type = 'steady_tempo'
            tempo_desc = 'Consistent forward pressure'
        
        analysis = {
            'interval': (start_frame, end_frame),
            'duration_frames': end_frame - start_frame + 1,
            'duration_seconds': (end_frame - start_frame + 1) / self.fps,
            
            # Attack analysis
            'attack_info': attack_info,
            
            # Tempo analysis
            'tempo_type': tempo_type,
            'tempo_description': tempo_desc,
            'tempo_changes': tempo_changes,
            'has_micro_pauses': has_pauses,
            'speed_variation': speed_var,
            
            # Distance management
            'min_distance': min_distance,
            'avg_distance': avg_distance,
            'good_attack_distance': (min_distance is not None) and (self.distance_threshold is not None) and (min_distance < self.distance_threshold),
            'dangerous_close_frames': dangerous_frames,
            'missed_opportunities': len(dangerous_frames),
            
            # Tactical assessment
            'tactical_notes': []
        }
        
        # Add tactical notes
        if analysis['good_attack_distance'] and attack_info['has_attack']:
            analysis['tactical_notes'].append('Good distance management with attack execution')
        elif analysis['good_attack_distance'] and not attack_info['has_attack']:
            analysis['tactical_notes'].append('Missed opportunity - in range but no attack')
        elif not analysis['good_attack_distance'] and attack_info['has_attack']:
            analysis['tactical_notes'].append('Attack from suboptimal distance')
        
        if tempo_type == 'broken_tempo':
            analysis['tactical_notes'].append('Using tempo changes to create openings')
        elif tempo_type == 'steady_tempo' and attack_info['attack_type'] == 'simple_attack':
            analysis['tactical_notes'].append('Direct offensive pressure')
        
        return analysis
    
    def analyze_retreat_interval(self, retreat_interval, left_xdata_df, left_ydata_df,
                               right_xdata_df, right_ydata_df, opponent_advances,
                               opponent_pauses, opponent_launches, defender_extensions,
                               is_left_defending):
        """
        Comprehensive analysis of a retreat/defense interval.
        
        Identifies counter-opportunities by merging consecutive frames of close distance
        into single, meaningful intervals.
        
        Returns detailed defensive analysis
        """
        start_frame, end_frame = retreat_interval
        
        # Calculate distances
        mask = (left_xdata_df['Frame'] >= start_frame) & (left_xdata_df['Frame'] <= end_frame)
        interval_left_x = left_xdata_df[mask]
        interval_left_y = left_ydata_df[mask]
        interval_right_x = right_xdata_df[mask]
        interval_right_y = right_ydata_df[mask]
        
        distances = self.calculate_fencer_distance(
            interval_left_x, interval_right_x,
            interval_left_y, interval_right_y,
            interval_left_x['Frame'].values
        )
        
        counter_opportunities = []
        
        # 1. Check for opportunities during opponent pauses (discrete events)
        for pause_start, pause_end in opponent_pauses:
            if start_frame <= pause_start <= end_frame:
                defender_action = any(pause_start <= ext['start_frame'] <= pause_end for ext in defender_extensions)
                counter_opportunities.append({
                    'type': 'opponent_pause',
                    'start_frame': pause_start,
                    'end_frame': pause_end,
                    'action_taken': defender_action
                })

        # 2. Merge consecutive close-distance frames into single opportunity intervals
        close_distance_interval_start = None
        interval_frames = interval_left_x['Frame'].values

        for i, (d, frame) in enumerate(zip(distances, interval_frames)):
            is_close = d < self.distance_threshold
            
            # Start of a new close-distance interval
            if is_close and close_distance_interval_start is None:
                close_distance_interval_start = frame
                
            # End of a close-distance interval
            elif not is_close and close_distance_interval_start is not None:
                interval_end_frame = interval_frames[i-1] # The last frame that was close
                
                # Check if the defender took action during this entire interval
                action_taken = any(close_distance_interval_start <= ext['start_frame'] <= interval_end_frame for ext in defender_extensions)
                
                counter_opportunities.append({
                    'type': 'close_distance',
                    'start_frame': close_distance_interval_start,
                    'end_frame': interval_end_frame,
                    'action_taken': action_taken
                })
                
                # Reset for the next potential interval
                close_distance_interval_start = None

        # Edge Case: Handle an interval that's still open at the end of the retreat
        if close_distance_interval_start is not None:
            interval_end_frame = interval_frames[-1]
            action_taken = any(close_distance_interval_start <= ext['start_frame'] <= interval_end_frame for ext in defender_extensions)
            counter_opportunities.append({
                'type': 'close_distance',
                'start_frame': close_distance_interval_start,
                'end_frame': interval_end_frame,
                'action_taken': action_taken
            })
        
        # Check response to opponent launches
        launch_responses = []
        for launch in opponent_launches:
            if start_frame <= launch['start_frame'] <= end_frame:
                # Calculate acceleration during and after launch
                launch_frame = int(launch['start_frame'])
                response_frames = range(launch_frame, min(launch_frame + 10, int(end_frame)))
                
                defender_data = left_xdata_df if is_left_defending else right_xdata_df
                
                response_mask = defender_data['Frame'].isin(response_frames)
                response_positions = defender_data[response_mask]['16'].values
                
                if len(response_positions) >= 2:
                    response_velocity = np.gradient(response_positions)
                    response_acceleration = np.gradient(response_velocity)
                    
                    # Negative acceleration for left defender = moving left (retreating)
                    # Positive acceleration for right defender = moving right (retreating)
                    pulled_distance = np.mean(response_acceleration) < -0.1 if is_left_defending else np.mean(response_acceleration) > 0.1
                else:
                    pulled_distance = False
                
                launch_responses.append({
                    'launch_frame': launch_frame,
                    'pulled_distance': pulled_distance
                })
        
        # Calculate defensive metrics (use None when distances unavailable)
        min_distance = np.min(distances) if len(distances) > 0 else None
        avg_distance = np.mean(distances) if len(distances) > 0 else None
        distance_variance = np.var(distances) if len(distances) > 1 else 0
        
        # Assess defensive quality
        maintained_safe_distance = (min_distance is not None) and (self.distance_threshold is not None) and (min_distance >= self.distance_threshold * 0.8)
        consistent_spacing = distance_variance < 0.7
        
        analysis = {
            'interval': (start_frame, end_frame),
            'duration_frames': end_frame - start_frame + 1,
            'duration_seconds': (end_frame - start_frame + 1) / self.fps,
            
            # Distance management
            'min_distance': min_distance,
            'avg_distance': avg_distance,
            'distance_variance': distance_variance,
            'maintained_safe_distance': maintained_safe_distance,
            'consistent_spacing': consistent_spacing,
            
            # Counter opportunities
            'counter_opportunities': counter_opportunities,
            'opportunities_taken': sum(1 for opp in counter_opportunities if opp['action_taken']),
            'opportunities_missed': sum(1 for opp in counter_opportunities if not opp['action_taken']),
            
            # Launch responses
            'launch_responses': launch_responses,
            'successful_distance_pulls': sum(1 for resp in launch_responses if resp['pulled_distance']),
            
            # Defensive assessment
            'defensive_quality': 'good' if maintained_safe_distance or consistent_spacing else 'poor',
            'tactical_notes': []
        }
        
        # Add tactical notes
        if analysis['opportunities_missed'] > 0:
            analysis['tactical_notes'].append(f"Missed {analysis['opportunities_missed']} counter-attack opportunities")
        
        if maintained_safe_distance:
            analysis['tactical_notes'].append("Good distance management throughout defense")
        else:
            analysis['tactical_notes'].append("Allowed opponent to get too close")
        
        if len(launch_responses) > 0 and analysis['successful_distance_pulls'] == 0:
            analysis['tactical_notes'].append("Failed to pull distance against launches")
        elif analysis['successful_distance_pulls'] > 0:
            analysis['tactical_notes'].append("Successfully pulled distance against attacks")
        
        return analysis

    def generate_interval_summary(self, advance_analyses, retreat_analyses, fencer_name):
        """
        Generate a comprehensive summary of interval analyses
        """
        summary = {
            'fencer': fencer_name,
            'total_advances': len(advance_analyses),
            'total_retreats': len(retreat_analyses),
            
            # Attack summary
            'attacks': {
                'total': sum(1 for a in advance_analyses if a['attack_info']['has_attack']),
                'simple': sum(1 for a in advance_analyses if a['attack_info']['attack_type'] == 'simple_attack'),
                'compound': sum(1 for a in advance_analyses if a['attack_info']['attack_type'] == 'compound_attack'),
                'holding': sum(1 for a in advance_analyses if a['attack_info']['attack_type'] == 'holding_attack'),
                'preparations': sum(1 for a in advance_analyses if 'preparation' in a['attack_info']['attack_type'])
            },
            
            # Tempo summary
            'tempo': {
                'steady': sum(1 for a in advance_analyses if a['tempo_type'] == 'steady_tempo'),
                'variable': sum(1 for a in advance_analyses if a['tempo_type'] == 'variable_tempo'),
                'broken': sum(1 for a in advance_analyses if a['tempo_type'] == 'broken_tempo')
            },
            
            # Distance management
            'distance': {
                'good_attack_distances': sum(1 for a in advance_analyses if a['good_attack_distance']),
                'missed_opportunities': sum(a['missed_opportunities'] for a in advance_analyses),
                'avg_min_distance': np.mean([a['min_distance'] for a in advance_analyses]) if advance_analyses else 0
            },
            
            # Defense summary
            'defense': {
                'good_distance_management': sum(1 for r in retreat_analyses if r['defensive_quality'] == 'good'),
                'counter_opportunities': sum(len(r['counter_opportunities']) for r in retreat_analyses),
                'counters_executed': sum(r['opportunities_taken'] for r in retreat_analyses),
                'successful_pulls': sum(r['successful_distance_pulls'] for r in retreat_analyses)
            }
        }
        
        return summary


def analyze_all_intervals(left_xdata_new_df, left_ydata_new_df, right_xdata_new_df, right_ydata_new_df,
                          left_advance, left_retreat, right_advance, right_retreat,
                          left_pause, right_pause, left_arm_extensions, right_arm_extensions,
                          left_launches, right_launches, fps=30, min_pause_frames=10):
    """
    Main function to analyze all intervals for both fencers
    
    Returns comprehensive analysis results
    """
    analyzer = FencingIntervalAnalyzer(fps=fps, min_pause_frames=min_pause_frames)
    
    # Analyze left fencer advances (left attacking, right defending)
    left_advance_analyses = []
    for interval in left_advance:
        analysis = analyzer.analyze_advance_interval(
            interval, left_xdata_new_df, left_ydata_new_df,
            right_xdata_new_df, right_ydata_new_df,
            left_arm_extensions, left_launches, is_left_attacking=True
        )
        left_advance_analyses.append(analysis)
    
    # Analyze right fencer advances (right attacking, left defending)
    right_advance_analyses = []
    for interval in right_advance:
        analysis = analyzer.analyze_advance_interval(
            interval, left_xdata_new_df, left_ydata_new_df,
            right_xdata_new_df, right_ydata_new_df,
            right_arm_extensions, right_launches, is_left_attacking=False
        )
        right_advance_analyses.append(analysis)
    
    # Analyze left fencer retreats (left defending, right attacking)
    left_retreat_analyses = []
    for interval in left_retreat:
        analysis = analyzer.analyze_retreat_interval(
            interval, left_xdata_new_df, left_ydata_new_df,
            right_xdata_new_df, right_ydata_new_df,
            right_advance, right_pause, right_launches,
            left_arm_extensions, is_left_defending=True
        )
        left_retreat_analyses.append(analysis)
    
    # Analyze right fencer retreats (right defending, left attacking)
    right_retreat_analyses = []
    for interval in right_retreat:
        analysis = analyzer.analyze_retreat_interval(
            interval, left_xdata_new_df, left_ydata_new_df,
            right_xdata_new_df, right_ydata_new_df,
            left_advance, left_pause, left_launches,
            right_arm_extensions, is_left_defending=False
        )
        right_retreat_analyses.append(analysis)
    
    # Generate summaries
    left_summary = analyzer.generate_interval_summary(
        left_advance_analyses, left_retreat_analyses, "Left Fencer"
    )
    right_summary = analyzer.generate_interval_summary(
        right_advance_analyses, right_retreat_analyses, "Right Fencer"
    )
    
    return {
        'left_fencer': {
            'advance_analyses': left_advance_analyses,
            'retreat_analyses': left_retreat_analyses,
            'summary': left_summary
        },
        'right_fencer': {
            'advance_analyses': right_advance_analyses,
            'retreat_analyses': right_retreat_analyses,
            'summary': right_summary
        }
    }


def print_interval_analysis_report(analysis_results):
    """
    Print a formatted report of interval analyses
    """
    for fencer in ['left_fencer', 'right_fencer']:
        print(f"\n{'='*60}")
        print(f"{fencer.replace('_', ' ').title()} Interval Analysis")
        print(f"{'='*60}")
        
        data = analysis_results[fencer]
        summary = data['summary']
        
        # Print summary statistics
        print(f"\nSummary Statistics:")
        print(f"  Total Advances: {summary['total_advances']}")
        print(f"  Total Retreats: {summary['total_retreats']}")
        
        print(f"\nAttack Breakdown:")
        print(f"  Total Attacks: {summary['attacks']['total']}")
        print(f"  - Simple: {summary['attacks']['simple']}")
        print(f"  - Compound: {summary['attacks']['compound']}")
        print(f"  - Holding: {summary['attacks']['holding']}")
        print(f"  - Preparations: {summary['attacks']['preparations']}")
        
        print(f"\nTempo Usage:")
        print(f"  Steady: {summary['tempo']['steady']}")
        print(f"  Variable: {summary['tempo']['variable']}")
        print(f"  Broken: {summary['tempo']['broken']}")
        
        print(f"\nDistance Management:")
        print(f"  Good Attack Distances: {summary['distance']['good_attack_distances']}")
        print(f"  Missed Opportunities: {summary['distance']['missed_opportunities']}")
        print(f"  Average Min Distance: {summary['distance']['avg_min_distance']:.2f}")
        
        print(f"\nDefensive Performance:")
        print(f"  Good Distance Management: {summary['defense']['good_distance_management']}")
        print(f"  Counter Opportunities: {summary['defense']['counter_opportunities']}")
        print(f"  Counters Executed: {summary['defense']['counters_executed']}")
        print(f"  Successful Distance Pulls: {summary['defense']['successful_pulls']}")
        
        # Print detailed advance analyses
        print(f"\nDetailed Advance Analyses:")
        for i, analysis in enumerate(data['advance_analyses']):
            print(f"\n  Advance {i+1} (frames {analysis['interval'][0]}-{analysis['interval'][1]}):")
            print(f"    Attack Type: {analysis['attack_info']['attack_type']}")
            print(f"    Tempo: {analysis['tempo_type']} (Pauses: {analysis['has_micro_pauses']})") # Added for clarity
            print(f"    Min Distance: {analysis['min_distance']:.2f}")
            print(f"    Tactical Notes: {'; '.join(analysis['tactical_notes'])}")
        
        # Print detailed retreat analyses
        print(f"\nDetailed Retreat Analyses:")
        for i, analysis in enumerate(data['retreat_analyses']):
            print(f"\n  Retreat {i+1} (frames {analysis['interval'][0]}-{analysis['interval'][1]}):")
            print(f"    Defensive Quality: {analysis['defensive_quality']}")
            print(f"    Counter Opportunities: {len(analysis['counter_opportunities'])}")
            print(f"    Opportunities Taken: {analysis['opportunities_taken']}")
            print(f"    Tactical Notes: {'; '.join(analysis['tactical_notes'])}")


def analyze_fencing_bout(video_id, frame_range, left_data, right_data, total_frames, video_angle, fps=30, frame_numbers=None, match_idx=None):
    try:
        logging.debug(f"Analyzing bout: video_id={video_id}, frames={frame_range}, total_frames={total_frames}")
        
        # Convert intervals to seconds
        left_adv_sec = convert_intervals_to_seconds(left_data['advance'], fps)
        left_pause_retreat_sec = convert_intervals_to_seconds(left_data['pause'], fps)
        right_adv_sec = convert_intervals_to_seconds(right_data['advance'], fps)
        right_pause_retreat_sec = convert_intervals_to_seconds(right_data['pause'], fps)
        left_arm_extensions_sec = convert_intervals_to_seconds(left_data['arm_extensions'], fps)
        right_arm_extensions_sec = convert_intervals_to_seconds(right_data['arm_extensions'], fps)
        
        # Latest pause/retreat end times
        left_latest_end = left_data['latest_pause_retreat_end']
        right_latest_end = right_data['latest_pause_retreat_end']
        left_latest_end_sec = round(left_latest_end / fps, 2) if left_latest_end != -1 else -1
        right_latest_end_sec = round(right_latest_end / fps, 2) if right_latest_end != -1 else -1
        
        # First step metrics - handle None values for safe formatting
        left_first_step = left_data['first_step']
        right_first_step = right_data['first_step']
        
        # Ensure all first_step values are not None for formatting
        for step_dict in [left_first_step, right_first_step]:
            if step_dict.get('init_time') is None:
                step_dict['init_time'] = 0.0
            if step_dict.get('velocity') is None:
                step_dict['velocity'] = 0.0
            if step_dict.get('acceleration') is None:
                step_dict['acceleration'] = 0.0
            if step_dict.get('momentum') is None:
                step_dict['momentum'] = 0.0
        
        # Ensure overall velocity/acceleration fields are not None
        for data_dict in [left_data, right_data]:
            if data_dict.get('overall_velocity') is None:
                data_dict['overall_velocity'] = 0.0
            if data_dict.get('overall_acceleration') is None:
                data_dict['overall_acceleration'] = 0.0
            if data_dict.get('attacking_velocity') is None:
                data_dict['attacking_velocity'] = 0.0
            if data_dict.get('attacking_acceleration') is None:
                data_dict['attacking_acceleration'] = 0.0
            if data_dict.get('launch_frame') is None:
                data_dict['launch_frame'] = 0
        
        # Interval metrics after latest pause/retreat
        use_in_box_rule = total_frames <= 80
        left_velocity_interval = left_data.get('velocity', 0.0) or 0.0
        right_velocity_interval = right_data.get('velocity', 0.0) or 0.0
        left_acceleration_interval = left_data.get('acceleration', 0.0) or 0.0
        right_acceleration_interval = right_data.get('acceleration', 0.0) or 0.0
        left_arm_in_interval = len([s for s, e in left_arm_extensions_sec if s >= left_latest_end_sec])
        right_arm_in_interval = len([s for s, e in right_arm_extensions_sec if s >= right_latest_end_sec])
        
        if use_in_box_rule and (left_latest_end != -1 or right_latest_end != -1):
            latest_end = max(left_latest_end if left_latest_end != -1 else float('-inf'), 
                             right_latest_end if right_latest_end != -1 else float('-inf'))
            latest_end_sec = round(latest_end / fps, 2)
            end_of_bout = frame_range[1]
            left_interval_df = pd.DataFrame({'16': left_data['front_foot_x'], 'Frame': frame_numbers})
            right_interval_df = pd.DataFrame({'16': right_data['front_foot_x'], 'Frame': frame_numbers})
            left_velocity_interval, left_acceleration_interval = calculate_velocity_and_acceleration_interval(
                left_interval_df, True, latest_end, end_of_bout, video_angle, fps=fps
            )
            right_velocity_interval, right_acceleration_interval = calculate_velocity_and_acceleration_interval(
                right_interval_df, False, latest_end, end_of_bout, video_angle, fps=fps
            )
        
        # Get bout result
        upload_id = int(video_id.split('_')[1]) if 'upload_' in video_id else None
        bout_result = None
        result_text = "Not specified"
        if upload_id and match_idx:
            bout = Bout.query.filter_by(upload_id=upload_id, match_idx=match_idx).first()
            if bout:
                bout_result = bout.result
                if bout_result == 'skip':
                    result_text = "Skipped"
                elif bout_result == 'left':
                    result_text = "Left wins"
                elif bout_result == 'right':
                    result_text = "Right wins"
        
        # Prepare GPT prompt
        prompt = f"""
You are an AI fencing analyst specializing in sabre, tasked with analyzing a bout and providing detailed strategic analysis based on data, focusing on understanding each fencer's strategy, including when they take initiative, observe, or defend, based on their movements, initial step metrics, velocity, acceleration, pauses, arm extensions, and lunges. The bout result is known ({result_text}), avoid referee terminology or declaring winners, the goal is to analyze strategy not judge outcomes. Analysis should explain why fencers made specific decisions (e.g., pause to induce, accelerate to attack).

**Input Data**:
- **Video ID**: {video_id}
- **Match Index**: {match_idx}
- **Result**: {result_text}
- **Frame Range**: {frame_range[0] / fps:.2f} to {frame_range[1] / fps:.2f} seconds
- **Total Frames**: {total_frames}
- **Video Angle**: {video_angle}
- **FPS**: {fps}

**Left Fencer**:
- **Initial Step Metrics**:
  - Start time: {left_first_step['init_time']:.2f} seconds
  - Velocity: {left_first_step['velocity']:.2f}
  - Acceleration: {left_first_step['acceleration']:.2f}
  - Momentum: {left_first_step['momentum']:.2f}
- **Advance intervals**: {left_adv_sec} (seconds)
- **Pause/retreat intervals**: {left_pause_retreat_sec} (seconds)
- **Arm extensions**: {left_arm_extensions_sec} (seconds)
- **Has lunge**: {left_data['has_launch']} (if yes, at frame {left_data['launch_frame']}, {left_data['launch_frame'] / fps:.2f} seconds)
- **Latest pause/retreat end**: {left_latest_end_sec} seconds
- **Overall velocity**: {left_data['overall_velocity']:.2f}
- **Overall acceleration**: {left_data['overall_acceleration']:.2f}
- **Attacking velocity**: {left_data['attacking_velocity']:.2f}
- **Attacking acceleration**: {left_data['attacking_acceleration']:.2f}
- **Post-pause interval velocity**: {left_velocity_interval:.2f}
- **Post-pause interval acceleration**: {left_acceleration_interval:.2f}
- **Post-pause arm extensions count**: {left_arm_in_interval}
  - **Interval analysis**: {left_data.get('interval_analysis', {})}

**Right Fencer**:
- **Initial Step Metrics**:
  - Start time: {right_first_step['init_time']:.2f} seconds
  - Velocity: {right_first_step['velocity']:.2f}
  - Acceleration: {right_first_step['acceleration']:.2f}
  - Momentum: {right_first_step['momentum']:.2f}
- **Advance intervals**: {right_adv_sec} (seconds)
- **Pause/retreat intervals**: {right_pause_retreat_sec} (seconds)
- **Arm extensions**: {right_arm_extensions_sec} (seconds)
- **Has lunge**: {right_data['has_launch']} (if yes, at frame {right_data['launch_frame']}, {right_data['launch_frame'] / fps:.2f} seconds)
- **Latest pause/retreat end**: {right_latest_end_sec} seconds
- **Overall velocity**: {right_data['overall_velocity']:.2f}
- **Overall acceleration**: {right_data['overall_acceleration']:.2f}
- **Attacking velocity**: {right_data['attacking_velocity']:.2f}
- **Attacking acceleration**: {right_data['attacking_acceleration']:.2f}
- **Post-pause interval velocity**: {right_velocity_interval:.2f}
- **Post-pause interval acceleration**: {right_acceleration_interval:.2f}
- **Post-pause arm extensions count**: {right_arm_in_interval}
  - **Interval analysis**: {right_data.get('interval_analysis', {})}

**Output Format (English)**:
1. **Bout Overview**:
   - Summarize bout context: duration, frame count, video angle, and result.
   - Describe overall strategic flow (e.g., aggressive exchanges, cautious probing, defensive interactions).
   - Highlight key moments (e.g., lunges, significant pauses, rapid advances) with their timing in seconds.

2. **Fencer Strategy & Decision-Making**:
   - For each fencer (left and right):
     - **Initial Step Analysis**: Discuss start time, velocity, acceleration, and momentum. Categorize as aggressive (fast, high metrics), observational (slow, low metrics), or defensive (minimal movement). Explain reasoning (e.g., fast step to take control, slow step to assess opponent).
     - **Movement Patterns**: Describe advance, pause/retreat intervals and their timing. Explain strategic intent (e.g., advance to close distance, pause to induce, retreat to reset).
     - **Arm Extensions**: Analyze timing and frequency of arm extensions. Discuss whether they indicate probing, feints, or attack preparation, and why (e.g., early extensions to test opponent, late extensions to exploit openings).
     - **Lunges**: If present, describe timing and context (e.g., after pause, during advance). Explain strategic significance (e.g., exploit opponent pause, counter-attack to advance).
     - **Velocity & Acceleration**: Analyze both overall and attacking metrics. Overall velocity/acceleration represents movement throughout the entire bout, while attacking velocity/acceleration represents movement during launch/attack phases only. Discuss strategic implications of acceleration (rapid changes) vs deceleration (slowing down) (e.g., accelerate to attack, decelerate to defend). Compare overall vs attacking metrics to understand tactical approach.
   - Provide insights into why fencers made these decisions, relating to bout result (e.g., pauses to disrupt opponent rhythm, acceleration to overwhelm opponent).

3. **Comparative Analysis**:
   - Compare fencer strategies: who was more aggressive, cautious, or defensive? How did their initial step metrics, movement patterns, and arm extensions differ?
   - Discuss interactions: how did one fencer's movements affect the other (e.g., left's advance forced right's retreat, right's pause induced left's lunge)?
   - Highlight strategic adjustments (e.g., shift from observation to attack, response to opponent's aggression), considering bout result.

4. **Summary Table**:
   - Compare key metrics:
     - Initial step start time (seconds)
     - Initial step velocity
     - Initial step acceleration
     - Initial step momentum
     - Overall velocity (entire bout movement)
     - Overall acceleration (entire bout movement)
     - Attacking velocity (launch/attack phases only)
     - Attacking acceleration (launch/attack phases only)
     - Arm extension count
     - Latest arm extension end (seconds, "N/A" if none)
     - Lunge time (seconds, "N/A" if none)
   - Indicate which fencer had advantage in each metric (or "None" if equal), relating to bout result.

**Guidelines**:
- Use precise timing in seconds for intervals and events.
- Explain strategic decisions with data-driven insights (e.g., "Left fencer paused at 2.5s to induce right's advance, then lunged at 3.2s").
- Avoid referee terminology (e.g., right of way, winner). Focus on analysis not judgment.
- Explain why fencers chose specific actions (e.g., "Right's slow initial step indicates observation, waiting for left to commit").
- Only consider video angle for velocity/acceleration scaling, not for excluding data.
- Provide logically flowing narrative connecting metrics to behaviors.
- All content should be in English.
"""
        # Use English prompt directly (no translation to Chinese)
        english_prompt = prompt  # Keep original English prompt

        # Exponential backoff with respect to 429 quota errors
        import time as _time
        import random as _random
        max_attempts = 6
        base_delay = float(os.getenv('GEMINI_BACKOFF_BASE_S', '2.0'))
        max_delay = float(os.getenv('GEMINI_BACKOFF_MAX_S', '60'))

        from your_scripts.gemini_rest import generate_text as gemini_generate_text
        for attempt in range(max_attempts):
            try:
                text = gemini_generate_text(
                    english_prompt,
                    model=os.getenv('GEMINI_MODEL', 'gemini-2.5-flash-lite'),
                    temperature=0.2,
                    top_k=1,
                    top_p=0.8,
                    max_output_tokens=2048,
                    timeout_seconds=45,
                    max_attempts=max_attempts,
                    response_mime_type='text/plain',
                )
                return text
            except Exception as call_err:
                # Detect 429 quota exceeded and suggested retry delay if present
                err_text = str(call_err)
                is_quota = '429' in err_text or 'quota' in err_text.lower() or 'ResourceExhausted' in err_text
                retry_after = None
                # Parse "Please retry in Xs" if present
                try:
                    import re as _re
                    m = _re.search(r'retry in\s*([0-9]+\.?[0-9]*)s', err_text, flags=_re.IGNORECASE)
                    if m:
                        retry_after = float(m.group(1))
                except Exception:
                    retry_after = None

                if attempt < max_attempts - 1 and is_quota:
                    # Compute backoff with jitter; honor server-provided retry delay if larger
                    exp = min(max_delay, base_delay * (2 ** attempt))
                    delay = max(exp * (1 + 0.25 * _random.random()), retry_after or 0.0)
                    delay = min(delay, max_delay)
                    logging.warning(f"Gemini quota/backoff hit (attempt {attempt+1}/{max_attempts}); sleeping {delay:.2f}s before retry")
                    _time.sleep(delay)
                    continue
                # Non-quota or final attempt: raise
                raise
    except Exception as e:
        logging.error(f"Error in analyze_fencing_bout: {str(e)}\n{traceback.format_exc()}")
        raise

def load_match_data(match_idx, match_data_dir):
    """Load match CSV data."""
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
                    raise ValueError(f"Non-numeric data in column {col}")
        logging.info(f"Loaded match {match_idx} data from {match_dir}")
        return left_x_df, left_y_df, right_x_df, right_y_df
    except Exception as e:
        logging.error(f"Error loading match {match_idx} data: {str(e)}\n{traceback.format_exc()}")
        return None, None, None, None

def process_match(match_idx, start_frame, end_frame, match_data_dir, video_id, video_angle='unknown', fps=30, bout_result=None):
    logging.debug(f"Processing match {match_idx}: Frames {start_frame} to {end_frame}")
    left_x_df, left_y_df, right_x_df, right_y_df = load_match_data(match_idx, match_data_dir)
    if left_x_df is None:
        logging.warning(f"Skipping match {match_idx} due to data loading error")
        return None
    
    total_frames = end_frame - start_frame + 1
    
    # NEW: Use comprehensive movement detection
    leftIntervals = detect_fencing_movements(left_x_df, is_left_fencer=True, plot=False)
    rightIntervals = detect_fencing_movements(right_x_df, is_left_fencer=False, plot=False)
    
    # Handle cases where movement detection fails or returns None (e.g., too few frames)
    if leftIntervals is None:
        logging.warning(f"Left fencer movement detection failed for match {match_idx}, using empty intervals")
        leftIntervals = {'advance_intervals': [], 'retreat_intervals': [], 'pause_intervals': []}
    if rightIntervals is None:
        logging.warning(f"Right fencer movement detection failed for match {match_idx}, using empty intervals")
        rightIntervals = {'advance_intervals': [], 'retreat_intervals': [], 'pause_intervals': []}
    
    left_advance = leftIntervals['advance_intervals']
    left_retreat = leftIntervals['retreat_intervals']
    left_pause = leftIntervals['pause_intervals']
    right_advance = rightIntervals['advance_intervals']
    right_retreat = rightIntervals['retreat_intervals']
    right_pause = rightIntervals['pause_intervals']

    # Analyze launches within advance intervals
    launch_analysis = analyze_launches_for_both_fencers(
        left_x_df, left_y_df, 
        right_x_df, right_y_df,
        left_advance, right_advance, 
        fps
    )
    left_launches = launch_analysis['left_fencer']['launches']
    right_launches = launch_analysis['right_fencer']['launches']
    
    left_has_launch = len(left_launches) > 0
    left_launch_frame = left_launches[0]['start_frame'] if left_launches else -1
    left_attacking_velocity = np.mean([l['front_foot_avg_velocity'] for l in left_launches]) if left_launches else 0
    left_attacking_acceleration = np.mean([l['front_foot_avg_acceleration'] for l in left_launches]) if left_launches else 0
    
    right_has_launch = len(right_launches) > 0
    right_launch_frame = right_launches[0]['start_frame'] if right_launches else -1
    right_attacking_velocity = np.mean([l['front_foot_avg_velocity'] for l in right_launches]) if right_launches else 0
    right_attacking_acceleration = np.mean([l['front_foot_avg_acceleration'] for l in right_launches]) if right_launches else 0

    # Calculate overall velocity and acceleration for the entire bout
    left_overall_velocity, left_overall_acceleration = calculate_overall_velocity_acceleration(
        left_x_df, is_left_fencer=True, video_angle=video_angle, fps=fps
    )
    right_overall_velocity, right_overall_acceleration = calculate_overall_velocity_acceleration(
        right_x_df, is_left_fencer=False, video_angle=video_angle, fps=fps
    )

    # Enhanced arm extension detection
    extension_analysis = analyze_arm_extensions_for_both_fencers(
        left_x_df, left_y_df,
        right_x_df, right_y_df,
        left_advance, right_advance,
        fps,
        threshold_distance=0.35
    )
    left_extensions = extension_analysis['left_fencer']['extensions']
    right_extensions = extension_analysis['right_fencer']['extensions']

    # First step metrics
    left_first_step, right_first_step = compute_first_step_metrics(left_x_df, right_x_df, fps)

    # Prepare data for interval analysis
    left_arm_extension_dicts = [{'start_frame': e['start_frame'], 'end_frame': e['end_frame']} for e in left_extensions]
    right_arm_extension_dicts = [{'start_frame': e['start_frame'], 'end_frame': e['end_frame']} for e in right_extensions]

    interval_analysis_results = analyze_all_intervals(
        left_x_df, left_y_df,
        right_x_df, right_y_df,
        left_advance, left_retreat,
        right_advance, right_retreat,
        left_pause, right_pause,
        left_arm_extension_dicts, right_arm_extension_dicts,
        left_launches, right_launches,
        fps
    )

    # --- Start compatibility metric calculations ---
    
    def calculate_compatibility_metrics(side, intervals, extensions, has_launch, launch_frame, attacking_velocity, attacking_acceleration, total_frames, fps, first_step, front_foot_x):
        advance_intervals = intervals.get('advance_intervals', [])
        pause_intervals = intervals.get('pause_intervals', [])
        retreat_intervals = intervals.get('retreat_intervals', [])
        
        pause_retreat_filtered = sorted(pause_intervals + retreat_intervals)

        # Determine the last frame where the fencer was paused/retreating, independent of launches
        latest_end = -1
        if pause_retreat_filtered:
            # Take the maximum end frame across all pause/retreat intervals
            latest_end = max(p[1] for p in pause_retreat_filtered)

        advance_sec = convert_intervals_to_seconds(advance_intervals, fps)
        pause_sec = convert_intervals_to_seconds(pause_retreat_filtered, fps)
        
        arm_extension_intervals = [(e['start_frame'], e['end_frame']) for e in extensions]
        arm_extensions_sec = convert_intervals_to_seconds(arm_extension_intervals, fps)
        
        total_advance_frames = sum(end - start + 1 for start, end in advance_intervals)
        total_pause_frames = sum(end - start + 1 for start, end in pause_retreat_filtered)
        
        advance_ratio = total_advance_frames / total_frames if total_frames > 0 else 0
        pause_ratio = total_pause_frames / total_frames if total_frames > 0 else 0
        
        arm_extension_freq = len(extensions)
        avg_arm_extension_duration = np.mean([e['duration_seconds'] for e in extensions]) if extensions else 0
        
        # Compute attacking score using observed behaviors instead of placeholder success rate
        # Factors considered:
        # - advance_ratio: proportion of advancing time
        # - arm_extension_freq: presence/frequency of arm extensions (capped at 2 for scoring)
        # - has_launch: whether a launch occurred in the phase
        # - attacking_velocity: normalized to [0,1] by clipping to an expected range
        # - attacking_acceleration: normalized to [0,1] by clipping to an expected range

        extension_factor = float(np.clip(arm_extension_freq / 2.0, 0.0, 1.0))
        launch_factor = 1.0 if has_launch else 0.0
        vel_score = float(np.clip(attacking_velocity / 2.0, 0.0, 1.0))
        accel_score = float(np.clip(attacking_acceleration / 3.0, 0.0, 1.0))

        attacking_score = (
            0.25 * advance_ratio +
            0.25 * extension_factor +
            0.25 * launch_factor +
            0.15 * vel_score +
            0.10 * accel_score
        ) * 100.0
        
        # Use None for unavailable timings to ensure strict JSON (no Infinity/NaN)
        launch_promptness = None
        if has_launch and latest_end != -1 and launch_frame is not None and launch_frame > latest_end:
            launch_promptness = (launch_frame - latest_end) / fps
        
        first_pause_time = pause_sec[0][0] if pause_sec else None
        first_restart_time = pause_sec[0][1] if pause_sec else None
        post_pause_velocity = attacking_velocity # Simplified - using attacking velocity for compatibility

        return {
            "advance": advance_intervals,
            "pause": pause_retreat_filtered,
            "latest_pause_retreat_end": latest_end,
            "arm_extensions": arm_extension_intervals,
            "front_foot_x": front_foot_x.tolist(),
            "first_step": first_step,
            "advance_sec": advance_sec,
            "pause_sec": pause_sec,
            "arm_extensions_sec": arm_extensions_sec,
            "advance_ratio": advance_ratio,
            "pause_ratio": pause_ratio,
            "arm_extension_freq": arm_extension_freq,
            "avg_arm_extension_duration": avg_arm_extension_duration,
            "attacking_score": attacking_score,
            "launch_promptness": launch_promptness,
            "first_pause_time": first_pause_time,
            "first_restart_time": first_restart_time,
            "post_pause_velocity": post_pause_velocity
        }

    left_comp_metrics = calculate_compatibility_metrics('left', leftIntervals, left_extensions, left_has_launch, left_launch_frame, left_attacking_velocity, left_attacking_acceleration, total_frames, fps, left_first_step, left_x_df['16'].values)
    right_comp_metrics = calculate_compatibility_metrics('right', rightIntervals, right_extensions, right_has_launch, right_launch_frame, right_attacking_velocity, right_attacking_acceleration, total_frames, fps, right_first_step, right_x_df['16'].values)
    # --- End compatibility metric calculations ---

    # Consolidate all data for each fencer
    left_data = {
        **left_comp_metrics, # Unpack all compatibility metrics
        'has_launch': left_has_launch,
        'launch_frame': left_launch_frame,
        # Keep legacy 'velocity' and 'acceleration' as overall metrics for backward compatibility
        'velocity': float(left_overall_velocity),
        'acceleration': float(left_overall_acceleration),
        # Add specific attacking and overall metrics
        'attacking_velocity': float(left_attacking_velocity),
        'attacking_acceleration': float(left_attacking_acceleration),
        'overall_velocity': float(left_overall_velocity),
        'overall_acceleration': float(left_overall_acceleration),
        'interval_analysis': interval_analysis_results.get('left_fencer', {}),
        'movement_data': {
            'advance_intervals': left_advance,
            'pause_intervals': left_pause,
            'retreat_intervals': left_retreat,
        },
        'summary_metrics': {
            'has_launch': left_has_launch,
            'launch_frame': left_launch_frame,
            'avg_velocity': left_attacking_velocity,  # Keep as attacking for compatibility
            'avg_acceleration': left_attacking_acceleration,  # Keep as attacking for compatibility
            'attacking_velocity': left_attacking_velocity,
            'attacking_acceleration': left_attacking_acceleration,
            'overall_velocity': left_overall_velocity,
            'overall_acceleration': left_overall_acceleration,
        },
        'launches': left_launches,
        'extensions': left_extensions,
    }

    right_data = {
        **right_comp_metrics, # Unpack all compatibility metrics
        'has_launch': right_has_launch,
        'launch_frame': right_launch_frame,
        # Keep legacy 'velocity' and 'acceleration' as overall metrics for backward compatibility
        'velocity': float(right_overall_velocity),
        'acceleration': float(right_overall_acceleration),
        # Add specific attacking and overall metrics
        'attacking_velocity': float(right_attacking_velocity),
        'attacking_acceleration': float(right_attacking_acceleration),
        'overall_velocity': float(right_overall_velocity),
        'overall_acceleration': float(right_overall_acceleration),
        'interval_analysis': interval_analysis_results.get('right_fencer', {}),
        'movement_data': {
            'advance_intervals': right_advance,
            'pause_intervals': right_pause,
            'retreat_intervals': right_retreat,
        },
        'summary_metrics': {
            'has_launch': right_has_launch,
            'launch_frame': right_launch_frame,
            'avg_velocity': right_attacking_velocity,  # Keep as attacking for compatibility
            'avg_acceleration': right_attacking_acceleration,  # Keep as attacking for compatibility
            'attacking_velocity': right_attacking_velocity,
            'attacking_acceleration': right_attacking_acceleration,
            'overall_velocity': right_overall_velocity,
            'overall_acceleration': right_overall_acceleration,
        },
        'launches': right_launches,
        'extensions': right_extensions,
    }
    
    frame_numbers = left_x_df['Frame'].tolist() if not left_x_df.empty else []
    
    # Generate enhanced GPT analysis with rich data
    gpt_analysis = analyze_fencing_bout(f"match_{match_idx}", (start_frame, end_frame), left_data, right_data, total_frames, video_angle, fps, frame_numbers, match_idx)
    
    # Calculate comprehensive bout statistics
    bout_statistics = {
        'total_duration_seconds': total_frames / fps,
        'left_fencer_summary': left_data['summary_metrics'],
        'right_fencer_summary': right_data['summary_metrics'],
        'left_interval_summary': left_data['interval_analysis'].get('summary', {}),
        'right_interval_summary': right_data['interval_analysis'].get('summary', {}),
    }
    
    # Import judge_bout_winner function
    from your_scripts.fencer_analysis import judge_bout_winner
    
    # Determine final winner
    if bout_result and bout_result.lower() not in ['skip', 'undetermined']:
        final_winner = bout_result.lower()
        winner_source = 'user'
        ai_judgment = None
    else:
        # Use AI judgment when result is skip or not provided
        temp_bout_data = {
            'frame_range': (start_frame, end_frame),
            'left_data': left_data,
            'right_data': right_data,
            'video_angle': video_angle
        }
        ai_judgment = judge_bout_winner(temp_bout_data, fps)
        final_winner = ai_judgment['winner'] if ai_judgment else 'undetermined'
        winner_source = 'ai'
    
    # Determine bout categories for each fencer using new pairwise classification
    total_bout_frames = end_frame - start_frame + 1
    left_category, right_category = classify_bout_categories(
        left_data,
        right_data,
        total_bout_frames,
        fps
    )
    
    # Determine overall bout type - use the more aggressive category
    if left_category == 'attack' or right_category == 'attack':
        bout_type = 'attack'
    elif left_category == 'defense' or right_category == 'defense':
        bout_type = 'defense'
    else:
        bout_type = 'in_box'
    
    # Determine first/second intention based on velocity threshold of 1
    left_first_step_velocity = left_data['first_step']['velocity']
    right_first_step_velocity = right_data['first_step']['velocity']
    
    left_intention = 'first_intention' if abs(left_first_step_velocity) > 1.0 else 'second_intention'
    right_intention = 'first_intention' if abs(right_first_step_velocity) > 1.0 else 'second_intention'
    
    result = {
        'gpt_analysis': gpt_analysis,
        'left_data': left_data,
        'right_data': right_data,
        'frame_range': (start_frame, end_frame),
        'frame_numbers': frame_numbers,
        'fps': fps,
        'upload_id': int(video_id.split('_')[1]) if 'upload_' in video_id else None,
        'match_idx': match_idx,
        'bout_statistics': bout_statistics,
        'video_angle': video_angle,
        'analysis_version': 'comprehensive_v3.0',
        # New fields added
        'winner': final_winner,
        'winner_source': winner_source,
        'ai_judgment': ai_judgment,
        'bout_type': bout_type,
        'left_fencer_category': left_category,
        'right_fencer_category': right_category,
        'left_intention': left_intention,
        'right_intention': right_intention
    }
    
    output_dir = os.path.join(match_data_dir, '..', 'match_analysis')
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f"match_{match_idx}_analysis.json")
    try:
        result_serializable = convert_numpy_types(result)
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(result_serializable, f, ensure_ascii=False, indent=4)
        logging.info(f"Saved match {match_idx} analysis to {output_path}")
    except Exception as e:
        import traceback
        logging.error(f"Error saving match {match_idx} analysis: {str(e)}\n{traceback.format_exc()}")
    
    return result

def main(match_idx, start_frame, end_frame, match_data_dir, video_id, video_angle='unknown', bout_result=None, fps=30):
    """Main function to analyze a fencing bout."""
    return process_match(match_idx, start_frame, end_frame, match_data_dir, video_id, video_angle, fps, bout_result)
