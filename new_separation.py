#!/usr/bin/env python3
"""
Simplified Fencing Match Detection Script
Input: CSV files + video path
Output: Video clips for each match
Uses combined distance + jerk detection for sabre fencing
"""

import numpy as np
import pandas as pd
import os
import cv2
import logging
import traceback

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger()


def detect_and_extract_matches(left_xdata_path, left_ydata_path, right_xdata_path, right_ydata_path,
                                video_path,
                                output_dir='match_outputs',
                                fps=30,
                                static_velocity_threshold=0.01,
                                movement_velocity_threshold=0.03,
                                min_static_frames=15,
                                min_movement_frames=3,
                                min_frame_spacing=100,
                                min_match_duration=10):
    """
    Detect fencing matches and extract video clips.
    
    Args:
        left_xdata_path: Path to left fencer X coordinates CSV
        left_ydata_path: Path to left fencer Y coordinates CSV
        right_xdata_path: Path to right fencer X coordinates CSV
        right_ydata_path: Path to right fencer Y coordinates CSV
        video_path: Path to the video file
        output_dir: Directory to save all outputs (default: 'match_outputs')
        fps: Video frames per second (default: 30)
        
        Detection parameters:
        static_velocity_threshold: Max velocity for static detection (default: 0.01)
        movement_velocity_threshold: Min velocity for movement detection (default: 0.03)
        min_static_frames: Min consecutive static frames (default: 15)
        min_movement_frames: Min consecutive moving frames (default: 3)
        min_frame_spacing: Min frames between matches (default: 100)
        min_match_duration: Min frames for a valid match (default: 30, ~1 second at 30fps)
    
    Returns:
        Dictionary with:
        - 'matches': List of (start_frame, end_frame) tuples
        - 'video_clips': List of paths to generated video clips
        - 'extended_clips': List of paths to extended video clips (with padding)
    """
    
    print(f"\n{'='*70}")
    print(f"FENCING MATCH DETECTION AND EXTRACTION")
    print(f"{'='*70}")
    print(f"Video: {video_path}")
    print(f"Output directory: {output_dir}")
    print(f"{'='*70}\n")
    
    # Create output directories
    os.makedirs(output_dir, exist_ok=True)
    
    # Load CSV data
    print("Loading CSV data...")
    left_x_df = pd.read_csv(left_xdata_path)
    left_y_df = pd.read_csv(left_ydata_path)
    right_x_df = pd.read_csv(right_xdata_path)
    right_y_df = pd.read_csv(right_ydata_path)
    
    # Add frame numbers if not present
    if 'Frame' not in left_x_df.columns:
        left_x_df = left_x_df.reset_index(drop=True)
        left_y_df = left_y_df.reset_index(drop=True)
        right_x_df = right_x_df.reset_index(drop=True)
        right_y_df = right_y_df.reset_index(drop=True)
        
        left_x_df['Frame'] = range(len(left_x_df))
        left_y_df['Frame'] = range(len(left_y_df))
        right_x_df['Frame'] = range(len(right_x_df))
        right_y_df['Frame'] = range(len(right_y_df))
    
    num_frames = len(left_x_df)
    print(f"✓ Loaded {num_frames} frames of data\n")
    
    # ==================== CALCULATE GLOBAL DISTANCE THRESHOLD ====================
    print("Calculating global distance threshold from entire video...")
    global_distance = abs(right_x_df['16'] - left_x_df['16']).values
    global_distance_threshold = np.percentile(global_distance, 7)  # Closest 10% of entire video
    min_global_distance = np.min(global_distance)
    print(f"✓ Global distance threshold (10th percentile): {global_distance_threshold:.3f}")
    print(f"✓ Minimum distance in entire video: {min_global_distance:.3f}\n")
    
    # ==================== DETECT START FRAMES ====================
    print(f"{'='*70}")
    print("STEP 1: DETECTING START FRAMES")
    print(f"{'='*70}\n")
    
    start_frames = _detect_start_frames(
        left_x_df, right_x_df,
        static_velocity_threshold, movement_velocity_threshold,
        min_static_frames, min_movement_frames, min_frame_spacing
    )
    
    # ==================== DETECT END FRAMES ====================
    print(f"\n{'='*70}")
    print("STEP 2: DETECTING END FRAMES")
    print(f"{'='*70}")
    print(f"Processing {len(start_frames)} start frames...")
    print("🎯 Using combined distance + jerk detection")
    print()
    
    all_jerk_frames = _calculate_jerk_frames(left_x_df, right_x_df)
    matches = []
    
    for i, start_frame in enumerate(start_frames):
        print(f"\n📍 Analyzing start frame {start_frame} (candidate {i+1}/{len(start_frames)})...")
        
        next_start = start_frames[i + 1] if i + 1 < len(start_frames) else num_frames
        
        # Get slice for this match
        slice_indices = left_x_df[(left_x_df['Frame'] >= start_frame) & 
                                 (left_x_df['Frame'] < next_start)].index
        
        # Skip if no data in this range
        if len(slice_indices) == 0:
            print(f"   ⚠️  No data before next start frame {next_start} - ABANDONED")
            continue
        
        # Check if slice is too short
        if len(slice_indices) < min_match_duration:
            print(f"   ⚠️  Insufficient frames ({len(slice_indices)}) before next start - ABANDONED")
            continue
        
        # Create temporary slices
        left_x_slice = left_x_df.loc[slice_indices].reset_index(drop=True)
        right_x_slice = right_x_df.loc[slice_indices].reset_index(drop=True)
        left_y_slice = left_y_df.loc[slice_indices].reset_index(drop=True)
        right_y_slice = right_y_df.loc[slice_indices].reset_index(drop=True)
        
        left_x_slice['Frame'] = left_x_slice['Frame'] - start_frame
        right_x_slice['Frame'] = right_x_slice['Frame'] - start_frame
        
        # Filter jerk frames to this match
        slice_jerk_frames = [f - start_frame for f in all_jerk_frames 
                           if start_frame <= f < next_start]
        
        # Try to find end frame using combined distance + jerk detection
        end_frame_found = False
        end_frame_idx = None
        
        try:
            print(f"      🔍 Detecting end using distance + jerk...")
            end_frame_idx = _find_end_frame_combined(left_x_slice, right_x_slice, slice_jerk_frames, global_distance_threshold)
            
            # Validate that end frame is reasonable
            if end_frame_idx is not None and end_frame_idx >= 0:
                end_frame = start_frame + end_frame_idx
                match_duration = end_frame - start_frame + 1
                
                # Check minimum match duration
                if match_duration < min_match_duration:
                    print(f"   ⚠️  Match too short ({match_duration} frames < {min_match_duration}) - ABANDONED")
                # Check if end frame is before next start frame
                elif end_frame >= next_start - 1:
                    print(f"   ⚠️  End frame {end_frame} >= next start {next_start} - ABANDONED")
                else:
                    end_frame_found = True
                    print(f"   ✅ Valid match found: {match_duration} frames ({match_duration/fps:.1f}s)")
            else:
                print(f"   ⚠️  No valid end frame detected - ABANDONED")
                
        except Exception as e:
            logger.error(f"Error finding end frame for start {start_frame}: {e}")
            print(f"   ⚠️  Error during end frame detection - ABANDONED")
        
        # Only add match if we found a valid end frame
        if end_frame_found:
            matches.append((start_frame, end_frame))
            duration = end_frame - start_frame + 1
            print(f"   ✓ Match {len(matches)}: frames {start_frame:5d} to {end_frame:5d} ({duration:4d} frames, {duration/fps:.1f}s)")
    
    # ==================== EXTRACT VIDEO CLIPS ====================
    
    # Check if we have any valid matches
    if len(matches) == 0:
        print(f"\n{'='*70}")
        print("⚠️  WARNING: No valid matches found!")
        print("All start frames were abandoned because no valid end frames were detected.")
        print(f"{'='*70}\n")
        return {
            'matches': [],
            'video_clips': [],
            'extended_clips': [],
            'output_dir': output_dir
        }
    
    print(f"\n{'='*70}")
    print("STEP 3: EXTRACTING VIDEO CLIPS")
    print(f"{'='*70}\n")
    
    video_clips = []
    extended_clips = []
    
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Could not open video: {video_path}")
    
    video_fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_video_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    print(f"Video properties: {width}x{height} @ {video_fps:.1f} fps, {total_video_frames} frames\n")
    
    for i, (start_frame, end_frame) in enumerate(matches, 1):
        match_dir = os.path.join(output_dir, f"match_{i:02d}")
        os.makedirs(match_dir, exist_ok=True)
        
        # Extract standard clip
        clip_path = os.path.join(match_dir, f"match_{i:02d}.mp4")
        _extract_video_segment(cap, start_frame, end_frame, clip_path, fps)
        video_clips.append(clip_path)
        print(f"✓ Match {i}: {clip_path}")
        
        # Extract extended clip (with 1 second padding)
        padding = int(fps)
        extended_start = max(0, start_frame - padding)
        extended_end = min(total_video_frames - 1, end_frame + padding)
        extended_path = os.path.join(match_dir, f"match_{i:02d}_extended.mp4")
        _extract_video_segment(cap, extended_start, extended_end, extended_path, fps)
        extended_clips.append(extended_path)
        print(f"  └─ Extended: {extended_path}")
        print()
    
    cap.release()
    
    # ==================== SUMMARY ====================
    print(f"{'='*70}")
    print(f"✅ COMPLETE")
    print(f"{'='*70}")
    print(f"Detected {len(start_frames)} potential start frames")
    print(f"Found valid end frames for {len(matches)} matches")
    if len(start_frames) > len(matches):
        print(f"Abandoned {len(start_frames) - len(matches)} start frames (no valid end frame)")
    print(f"Output directory: {output_dir}")
    print(f"{'='*70}\n")
    
    return {
        'matches': matches,
        'video_clips': video_clips,
        'extended_clips': extended_clips,
        'output_dir': output_dir
    }


# ==================== HELPER FUNCTIONS ====================

def _detect_start_frames(left_x_df, right_x_df, static_threshold, movement_threshold,
                         min_static, min_movement, min_spacing):
    """Detect match start frames using velocity-based method with spike detection"""
    
    num_frames = len(left_x_df)
    
    # Get front foot positions
    left_foot = left_x_df['16'].values if '16' in left_x_df.columns else left_x_df.iloc[:, 16].values
    right_foot = right_x_df['16'].values if '16' in right_x_df.columns else right_x_df.iloc[:, 16].values
    
    # Calculate velocities
    left_vel = np.abs(np.diff(left_foot))
    right_vel = np.abs(np.diff(right_foot))
    left_vel = np.insert(left_vel, 0, 0)
    right_vel = np.insert(right_vel, 0, 0)
    
    # Smooth velocities
    window = 3
    left_vel_smooth = np.convolve(left_vel, np.ones(window)/window, mode='same')
    right_vel_smooth = np.convolve(right_vel, np.ones(window)/window, mode='same')
    combined_vel = np.maximum(left_vel_smooth, right_vel_smooth)
    
    # Calculate velocity acceleration (change in velocity)
    vel_acceleration = np.diff(combined_vel)
    vel_acceleration = np.insert(vel_acceleration, 0, 0)
    
    # Find static-to-movement transitions
    candidate_frames = []
    i = 0
    
    while i < num_frames:
        if combined_vel[i] <= static_threshold:
            # Count consecutive static frames
            static_count = 0
            j = i
            static_avg_vel = []
            
            while j < num_frames and combined_vel[j] <= static_threshold:
                static_avg_vel.append(combined_vel[j])
                static_count += 1
                j += 1
            
            if static_count >= min_static:
                baseline_vel = np.mean(static_avg_vel)
                print(f"  Static period: frames {i:5d} to {j-1:5d} ({static_count} frames, baseline vel={baseline_vel:.4f})")
                
                # Look for SIGNIFICANT velocity increase (spike detection)
                movement_found = False
                lookahead = 20
                
                # Find frame where velocity SIGNIFICANTLY exceeds baseline
                for look_frame in range(j, min(j + lookahead, num_frames)):
                    # Look for frame where velocity is much higher than baseline
                    if combined_vel[look_frame] >= movement_threshold:
                        # Found potential start - but let's find the PEAK velocity nearby
                        peak_frame = look_frame
                        peak_vel = combined_vel[look_frame]
                        
                        # Check next few frames for peak
                        for check_frame in range(look_frame, min(look_frame + 5, num_frames)):
                            if combined_vel[check_frame] > peak_vel:
                                peak_vel = combined_vel[check_frame]
                                peak_frame = check_frame
                        
                        # Verify this is sustained movement (not just noise)
                        sustained_movement = 0
                        for check_frame in range(peak_frame, min(peak_frame + min_movement, num_frames)):
                            if combined_vel[check_frame] >= movement_threshold * 0.7:  # 70% of threshold
                                sustained_movement += 1
                        
                        if sustained_movement >= min_movement:
                            # Find the frame where velocity really starts increasing
                            # Work backwards from peak to find the "takeoff" point
                            takeoff_frame = peak_frame
                            for back_frame in range(peak_frame, max(j - 1, look_frame - 3), -1):
                                if combined_vel[back_frame] < movement_threshold * 0.5:
                                    takeoff_frame = back_frame + 1
                                    break
                                takeoff_frame = back_frame
                            
                            candidate_frames.append(takeoff_frame)
                            print(f"    → Movement spike at frame {peak_frame} (vel={peak_vel:.4f})")
                            print(f"    → Takeoff frame: {takeoff_frame} (vel={combined_vel[takeoff_frame]:.4f})")
                            movement_found = True
                            break
                
                if not movement_found:
                    print(f"    → No movement detected")
                
                i = j
            else:
                i += 1
        else:
            i += 1
    
    # Filter by spacing
    if len(candidate_frames) == 0:
        return [0]
    
    start_frames = [candidate_frames[0]]
    for frame in candidate_frames[1:]:
        if frame - start_frames[-1] >= min_spacing:
            start_frames.append(frame)
    
    print(f"\nFiltered to {len(start_frames)} start frames: {start_frames}")
    return start_frames


def _calculate_jerk_frames(left_x_df, right_x_df):
    """Calculate jerk-based reference frames for end detection"""
    left_velocity = np.diff(left_x_df['16'].values)
    right_velocity = -np.diff(right_x_df['16'].values)
    left_acceleration = np.diff(left_velocity)
    right_acceleration = np.diff(right_velocity)
    left_jerk = np.diff(left_acceleration)
    right_jerk = np.diff(right_acceleration)
    
    min_left_jerk_indices = np.argsort(left_jerk)[:3]
    min_right_jerk_indices = np.argsort(right_jerk)[:3]
    min_left_accel_indices = np.argsort(left_acceleration)[:3]
    min_right_accel_indices = np.argsort(right_acceleration)[:3]
    
    all_jerk_frames = np.concatenate([
        left_x_df.iloc[min_left_jerk_indices]['Frame'].values,
        right_x_df.iloc[min_right_jerk_indices]['Frame'].values,
        left_x_df.iloc[min_left_accel_indices]['Frame'].values,
        right_x_df.iloc[min_right_accel_indices]['Frame'].values
    ])
    
    return all_jerk_frames


def _find_end_frame_combined(left_x_slice, right_x_slice, jerk_frames, global_distance_threshold):
    """
    Find end frame by combining distance and jerk signals.
    
    In sabre fencing, a touch occurs when:
    1. Fencers get CLOSE (minimum distance)
    2. Sudden STOP/HIT (high jerk/deceleration)
    
    This function finds the frame where both signals align.
    
    Args:
        global_distance_threshold: Distance threshold calculated from entire video (10th percentile)
    """
    
    if len(left_x_slice) < 10:
        return None
    
    print(f"      Analyzing {len(left_x_slice)} frames for end detection...")
    print(f"      Global distance threshold: {global_distance_threshold:.3f}")
    
    # Calculate distance between fencers
    distance = abs(right_x_slice['16'] - left_x_slice['16']).values
    
    # Calculate velocities
    left_vel = np.abs(np.diff(left_x_slice['16'].values))
    right_vel = np.abs(np.diff(right_x_slice['16'].values))
    left_vel = np.insert(left_vel, 0, 0)
    right_vel = np.insert(right_vel, 0, 0)
    
    # Calculate acceleration (for jerk)
    left_accel = np.diff(left_vel)
    right_accel = np.diff(right_vel)
    left_accel = np.insert(left_accel, 0, 0)
    right_accel = np.insert(right_accel, 0, 0)
    
    # Calculate jerk (rate of change of acceleration - indicates sudden stops)
    left_jerk = np.abs(np.diff(left_accel))
    right_jerk = np.abs(np.diff(right_accel))
    left_jerk = np.insert(left_jerk, 0, 0)
    right_jerk = np.insert(right_jerk, 0, 0)
    
    combined_jerk = left_jerk + right_jerk
    combined_vel = np.maximum(left_vel, right_vel)
    
    # Smooth signals
    window = 3
    distance_smooth = np.convolve(distance, np.ones(window)/window, mode='same')
    jerk_smooth = np.convolve(combined_jerk, np.ones(window)/window, mode='same')
    
    # ========== SIGNAL 1: FIND CLOSE APPROACH ==========
    # Use GLOBAL threshold - frames where distance is below global 10th percentile
    close_frames = np.where((distance_smooth < global_distance_threshold) | (distance_smooth < 0.7))[0]
    print (close_frames)

    
    if len(close_frames) > 0:
        min_distance_frame = distance_smooth.argmin()
        print(f"         Min distance in slice: {distance_smooth[min_distance_frame]:.3f} at frame {min_distance_frame}")
        print(f"         Close frames (< {global_distance_threshold:.3f}): {len(close_frames)} frames")
    else:
        print(f"         No close approach detected (threshold: {global_distance_threshold:.3f})")
        return None
    
    # ========== SIGNAL 2: FIND HIGH JERK EVENTS ==========
    # High jerk indicates sudden stop/deceleration (likely a hit)
    jerk_threshold = np.percentile(jerk_smooth, 80)  # Top 20% of jerk values
    high_jerk_frames = np.where(jerk_smooth > jerk_threshold)[0]
    
    if len(high_jerk_frames) > 0:
        max_jerk_frame = jerk_smooth.argmax()
        print(f"         Max jerk: {jerk_smooth[max_jerk_frame]:.4f} at frame {max_jerk_frame}")
        print(f"         Jerk threshold: {jerk_threshold:.4f}")
        print(f"         High jerk frames: {len(high_jerk_frames)} frames")
    else:
        print(f"         No high jerk detected")
        high_jerk_frames = []
    
    # ========== COMBINE SIGNALS TO FIND HIT FRAME ==========
    # Look for frames where BOTH distance is small AND jerk is high
    hit_candidates = []
    
    # Method 1: Find intersection of close frames and high jerk frames
    for close_frame in close_frames:
        # Check if there's a high jerk event nearby (within 10 frames)
        nearby_jerk = [j for j in high_jerk_frames if abs(j - close_frame) <= 10]
        if nearby_jerk:
            # Score based on how close and how much jerk
            score = (1.0 - distance_smooth[close_frame] / global_distance_threshold) + \
                   (jerk_smooth[close_frame] / jerk_threshold if jerk_threshold > 0 else 0)
            hit_candidates.append((close_frame, score, 'distance+jerk'))
            print(f"         Hit candidate at frame {close_frame} (score={score:.2f}, distance={distance_smooth[close_frame]:.3f}, jerk={jerk_smooth[close_frame]:.4f})")
    
    # Method 2: Check around minimum distance point
    search_window = 15
    min_dist_start = max(0, min_distance_frame - search_window)
    min_dist_end = min(len(jerk_smooth), min_distance_frame + search_window)
    
    for frame in range(min_dist_start, min_dist_end):
        if jerk_smooth[frame] > jerk_threshold * 0.7:  # 70% of threshold
            score = (1.0 - distance_smooth[frame] / global_distance_threshold) + \
                   (jerk_smooth[frame] / jerk_threshold if jerk_threshold > 0 else 0)
            hit_candidates.append((frame, score, 'min_distance_region'))
    
    # Method 3: Find the moment of maximum combined score
    # Create combined score: close distance + high jerk
    combined_score = np.zeros(len(distance_smooth))
    for i in range(len(combined_score)):
        dist_score = max(0, 1.0 - distance_smooth[i] / global_distance_threshold)
        jerk_score = jerk_smooth[i] / (np.mean(jerk_smooth) + 1e-6)
        combined_score[i] = dist_score * 0.6 + jerk_score * 0.4  # Weight distance more
    
    # Find peak combined score in first 2/3 of the sequence
    search_end = int(len(combined_score) * 0.67)
    if search_end > 20:
        peak_score_frame = combined_score[:search_end].argmax()
        if combined_score[peak_score_frame] > 0.5:  # Significant score
            hit_candidates.append((peak_score_frame, combined_score[peak_score_frame], 'combined_score'))
            print(f"         Peak combined score at frame {peak_score_frame} (score={combined_score[peak_score_frame]:.2f})")
    
    # ========== SELECT BEST END FRAME ==========
    if len(hit_candidates) == 0:
        # No clear hit detected - use minimum distance + buffer
        end_frame = min(min_distance_frame + 15, len(left_x_slice) - 1)
        print(f"         No clear hit, using min distance + 15 frames: {end_frame}")
        return end_frame
    
    # Remove duplicates and sort by score
    unique_candidates = {}
    for frame, score, method in hit_candidates:
        if frame not in unique_candidates or score > unique_candidates[frame][0]:
            unique_candidates[frame] = (score, method)
    
    # Get best candidate
    best_frame = max(unique_candidates.items(), key=lambda x: x[1][0])
    hit_frame = best_frame[0]
    hit_score = best_frame[1][0]
    hit_method = best_frame[1][1]
    
    # Add buffer after hit (fencers need time to separate)
    buffer_frames = 10
    end_frame = min(hit_frame + buffer_frames, len(left_x_slice) - 1)
    
    print(f"         ✓ Hit detected at frame {hit_frame} (method: {hit_method}, score: {hit_score:.2f})")
    print(f"         ✓ End frame (with buffer): {end_frame}")
    
    return end_frame


def _extract_video_segment(cap, start_frame, end_frame, output_path, fps):
    """Extract a video segment from start_frame to end_frame"""
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    # Use mp4v codec (more compatible)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    frames_written = 0
    for frame_idx in range(start_frame, end_frame + 1):
        ret, frame = cap.read()
        if not ret:
            break
        out.write(frame)
        frames_written += 1
    
    out.release()
    return frames_written


# ==================== MAIN FUNCTION ====================

def main(left_xdata_path, left_ydata_path, right_xdata_path, right_ydata_path, video_path,
         output_dir='match_outputs', fps=30, **kwargs):
    """
    Simple main function - just provide CSV paths and video path.
    
    Args:
        left_xdata_path: Path to left_xdata.csv
        left_ydata_path: Path to left_ydata.csv
        right_xdata_path: Path to right_xdata.csv
        right_ydata_path: Path to right_ydata.csv
        video_path: Path to video file
        output_dir: Where to save outputs (default: 'match_outputs')
        fps: Video FPS (default: 30)
        **kwargs: Optional detection parameters
    
    Returns:
        Dictionary with matches, video_clips, and extended_clips
    """
    return detect_and_extract_matches(
        left_xdata_path, left_ydata_path, right_xdata_path, right_ydata_path,
        video_path, output_dir, fps, **kwargs
    )


# ==================== USAGE EXAMPLE ====================

if __name__ == "__main__":
    # Example usage with combined distance + jerk detection
    result = main(
        left_xdata_path='t4_left_xdata.csv',
        left_ydata_path='t4_left_ydata.csv',
        right_xdata_path='t4_right_xdata.csv',
        right_ydata_path='t4_right_ydata.csv',
        video_path='t4.MP4',
        output_dir='match_outputs_3',
        fps=30
    )
    
    print("\n📊 Results Summary:")
    print(f"   Matches: {len(result['matches'])}")
    for i, (start, end) in enumerate(result['matches'], 1):
        print(f"   Match {i}: frames {start}-{end}")
    print(f"\n📁 Output directory: {result['output_dir']}")