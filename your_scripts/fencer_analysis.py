import numpy as np
import pandas as pd
import os
import json
import math
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter
import logging
import traceback
import argparse
from models import Bout

# Import our new analysis and plotting modules
from your_scripts.analysis_builders import extract_fencer_analysis_data
from your_scripts.plotting import save_all_plots
from your_scripts.bout_classification import classify_all_bouts
from your_scripts.touch_visualization import create_touch_category_charts, generate_touch_category_summary, create_inbox_analysis_charts
from your_scripts.inbox_analysis import integrate_inbox_with_touch_classification, process_inbox_bouts_from_analysis
from your_scripts.attack_comprehensive_analysis import process_attack_bouts_with_winners, create_comprehensive_attack_charts
from your_scripts.defense_comprehensive_analysis import process_defense_bouts_with_winners, create_comprehensive_defense_charts

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s',
    handlers=[
        logging.FileHandler('/workspace/Project/fencer_analysis.log'),
        logging.StreamHandler()
    ]
)

# Note: Gemini SDK usage removed in favor of REST client elsewhere


def load_bout_data(analysis_dir, match_data_dir):
    """Load bout analysis JSON files, CSVs, and bout results."""
    bout_data = []
    for file_name in os.listdir(analysis_dir):
        if file_name.endswith('_analysis.json'):
            match_idx = int(file_name.split('_')[1].split('.')[0])
            file_path = os.path.join(analysis_dir, file_name)
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                # Include match_idx and upload_id in data
                data['match_idx'] = match_idx
                # Fetch upload_id from JSON file or video_id
                upload_id = data.get('upload_id')  # Prefer direct upload_id from JSON
                if upload_id is None and 'video_id' in data and isinstance(data['video_id'], str) and data['video_id'].startswith('upload_'):
                    try:
                        upload_id = int(data['video_id'].split('_')[1])
                    except (IndexError, ValueError):
                        logging.warning(f"Invalid video_id format in {file_path}: {data.get('video_id', 'None')}")
                data['upload_id'] = upload_id
                data['video_angle'] = data.get('video_angle')
                # Load bout result from database if upload_id is valid
                if upload_id:
                    bout_result = Bout.query.filter_by(upload_id=upload_id, match_idx=match_idx).first()
                    data['bout_result'] = bout_result.result if bout_result else None
                else:
                    data['bout_result'] = None
                    logging.info(f"No valid upload_id for match {match_idx} in {file_path}, skipping Bout query")
                    
                # Load fencer analysis data for velocity metrics
                try:
                    fencer_analysis_dir = os.path.join(os.path.dirname(analysis_dir), 'fencer_analysis')
                    left_fencer_file = os.path.join(fencer_analysis_dir, 'fencer_Fencer_Left_analysis.json')
                    right_fencer_file = os.path.join(fencer_analysis_dir, 'fencer_Fencer_Right_analysis.json')
                    
                    if os.path.exists(left_fencer_file):
                        with open(left_fencer_file, 'r', encoding='utf-8') as f:
                            left_fencer_data = json.load(f)
                            # Find the bout data for this match_idx
                            for bout in left_fencer_data.get('bouts', []):
                                if bout.get('match_idx') == match_idx:
                                    data['left_data']['attack_analysis'] = bout['metrics'].get('attack_analysis', {})
                                    break
                    
                    if os.path.exists(right_fencer_file):
                        with open(right_fencer_file, 'r', encoding='utf-8') as f:
                            right_fencer_data = json.load(f)
                            # Find the bout data for this match_idx  
                            for bout in right_fencer_data.get('bouts', []):
                                if bout.get('match_idx') == match_idx:
                                    data['right_data']['attack_analysis'] = bout['metrics'].get('attack_analysis', {})
                                    break
                except Exception as e:
                    logging.warning(f"Error loading fencer analysis data for match {match_idx}: {str(e)}")
                
                match_dir = os.path.join(match_data_dir, f"match_{match_idx}")
                try:
                    data['left_x_df'] = pd.read_csv(os.path.join(match_dir, 'left_xdata.csv'))
                    data['left_y_df'] = pd.read_csv(os.path.join(match_dir, 'left_ydata.csv'))
                    data['right_x_df'] = pd.read_csv(os.path.join(match_dir, 'right_xdata.csv'))
                    data['right_y_df'] = pd.read_csv(os.path.join(match_dir, 'right_ydata.csv'))
                    
                    # Adapt new bout analysis data structure to fencer_analysis expectations
                    data = adapt_new_bout_data_structure(data)
                    
                    bout_data.append(data)
                except Exception as e:
                    logging.error(f"Error loading CSVs for match {match_idx}: {str(e)}\n{traceback.format_exc()}")
            except Exception as e:
                logging.error(f"Error loading {file_path}: {str(e)}\n{traceback.format_exc()}")
    bout_data.sort(key=lambda x: x['match_idx'])
    logging.info(f"Loaded {len(bout_data)} bout analysis files with results")
    return bout_data


def adapt_new_bout_data_structure(data):
    """
    Adapt the new bout analysis data structure to what fencer_analysis expects.
    This function extracts required metrics from the new structure and calculates 
    missing ones from available data.
    """
    fps = data.get('fps', 30)
    
    for side in ['left', 'right']:
        side_data = data.get(f'{side}_data', {})
        
        # Extract movement data from the new structure
        movement_data = side_data.get('movement_data', {})
        advance_intervals = movement_data.get('advance_intervals', [])
        pause_intervals = movement_data.get('pause_intervals', [])
        retreat_intervals = movement_data.get('retreat_intervals', [])
        
        # Convert intervals to the expected format if they're not already
        if advance_intervals and isinstance(advance_intervals[0], dict):
            advance_intervals = [(interval['start'], interval['end']) for interval in advance_intervals]
        if pause_intervals and isinstance(pause_intervals[0], dict):
            pause_intervals = [(interval['start'], interval['end']) for interval in pause_intervals]
        if retreat_intervals and isinstance(retreat_intervals[0], dict):
            retreat_intervals = [(interval['start'], interval['end']) for interval in retreat_intervals]
        
        # Ensure basic structure exists
        if 'advance' not in side_data:
            side_data['advance'] = advance_intervals
        if 'pause' not in side_data:
            # Combine pause and retreat intervals as fencer_analysis expects
            side_data['pause'] = sorted(pause_intervals + retreat_intervals)
        
        # Extract arm extensions from the new structure
        extensions = side_data.get('extensions', [])
        if extensions and isinstance(extensions[0], dict):
            # Convert from new detailed extension format to simple intervals
            arm_extension_intervals = [(ext['start_frame'], ext['end_frame']) for ext in extensions]
        else:
            arm_extension_intervals = extensions
        side_data['arm_extensions'] = arm_extension_intervals
        
        # Extract launch information
        launches = side_data.get('launches', [])
        if launches:
            # Find the latest launch frame
            latest_launch = max(launches, key=lambda x: x.get('start_frame', 0))
            side_data['has_launch'] = True
            side_data['launch_frame'] = latest_launch.get('start_frame', -1)
        else:
            side_data['has_launch'] = side_data.get('has_launch', False)
            side_data['launch_frame'] = side_data.get('launch_frame', -1)
        
        # Extract velocity and acceleration from summary_metrics
        summary_metrics = side_data.get('summary_metrics', {})
        side_data['velocity'] = summary_metrics.get('avg_velocity', side_data.get('velocity', 0))
        side_data['acceleration'] = summary_metrics.get('avg_acceleration', side_data.get('acceleration', 0))
        
        # Calculate latest_pause_retreat_end from intervals
        all_pause_retreat = sorted(pause_intervals + retreat_intervals)
        if all_pause_retreat:
            side_data['latest_pause_retreat_end'] = max(interval[1] for interval in all_pause_retreat)
        else:
            side_data['latest_pause_retreat_end'] = -1
        
        # Extract front_foot_x if available
        if 'front_foot_x' not in side_data and f'{side}_x_df' in data:
            # Assume keypoint 16 is front foot (this is common in pose estimation)
            x_df = data[f'{side}_x_df']
            if '16' in x_df.columns:
                side_data['front_foot_x'] = x_df['16'].values.tolist()
            else:
                side_data['front_foot_x'] = []
        
        # Create attack_analysis structure from launches and extensions for velocity extraction
        if 'attack_analysis' not in side_data:
            attack_analysis = {
                'all_launches': [],
                'all_extensions': []
            }
            
            # Extract launch metrics
            for launch in launches:
                launch_metrics = {
                    'front_foot_velocity': launch.get('front_foot_max_velocity', 0),
                    'hip_velocity': launch.get('front_hip_max_velocity', 0)
                }
                attack_analysis['all_launches'].append(launch_metrics)
            
            # Extract extension metrics  
            for extension in extensions:
                if isinstance(extension, dict):
                    extension_metrics = {
                        'arm_velocity': extension.get('max_velocity', 0)
                    }
                    attack_analysis['all_extensions'].append(extension_metrics)
            
            side_data['attack_analysis'] = attack_analysis
        
        # Ensure first_step exists - if not, create a default structure
        if 'first_step' not in side_data:
            side_data['first_step'] = {
                'init_time': float('inf'),
                'velocity': 0,
                'acceleration': 0,
                'is_fast': False
            }
    
    return data


def compute_first_step_metrics(left_x_df, right_x_df, fps=30, velocity_threshold=0.01, window_size=5):
    """Compute first step initiation time, velocity, acceleration, and momentum."""
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
    left_metrics = {'init_time': float('inf'), 'velocity': 0.0, 'acceleration': 0.0, 'momentum': 0.0}
    right_metrics = {'init_time': float('inf'), 'velocity': 0.0, 'acceleration': 0.0, 'momentum': 0.0}
    if left_first_step is not None:
        left_metrics['init_time'] = left_first_step / fps
        end_idx = min(left_first_step + 8, len(left_x_smooth) - 1)
        if end_idx > left_first_step:
            v = left_x_smooth[left_first_step:end_idx] - left_x_smooth[left_first_step-1:end_idx-1]
            left_metrics['velocity'] = np.mean(v) * fps
            a = np.diff(v) * fps * fps
            left_metrics['acceleration'] = np.mean(a) if len(a) > 0 else 0.0
            left_metrics['momentum'] = abs(left_metrics['velocity'])
    if right_first_step is not None:
        right_metrics['init_time'] = right_first_step / fps
        end_idx = min(right_first_step + 8, len(right_x_smooth) - 1)
        if end_idx > right_first_step:
            v = right_x_smooth[right_first_step:end_idx] - right_x_smooth[right_first_step-1:end_idx-1]
            right_metrics['velocity'] = -np.mean(v) * fps
            a = np.diff(v) * fps * fps
            right_metrics['acceleration'] = -np.mean(a) if len(a) > 0 else 0.0
            right_metrics['momentum'] = abs(right_metrics['velocity'])
    return left_metrics, right_metrics

def extract_velocity_metrics_from_attacks(bout_data):
    """Extract front foot velocity, hip velocity, and arm velocity from attack analysis."""
    for bout in bout_data:
        for side in ['left', 'right']:
            data = bout[f'{side}_data']
            
            # Check if velocities are already calculated and skip if so
            if all(key in data for key in ['avg_front_foot_velocity', 'avg_hip_velocity_launch', 'avg_arm_velocity_extension']):
                continue
            
            # Debug: Log available keys in data
            logging.info(f"Available keys in {side}_data: {list(data.keys())}")
            
            # Extract attack analysis if available
            attack_analysis = data.get('attack_analysis', {})
            
            # Initialize velocity lists
            front_foot_velocities = []
            hip_velocities = []
            arm_velocities = []
            
            if attack_analysis:
                # Extract from all_launches (front foot and hip velocities)
                all_launches = attack_analysis.get('all_launches', [])
                for launch in all_launches:
                    if 'front_foot_velocity' in launch:
                        front_foot_velocities.append(launch['front_foot_velocity'])
                    if 'hip_velocity' in launch:
                        hip_velocities.append(launch['hip_velocity'])
                
                # Extract from all_extensions (arm velocities)
                all_extensions = attack_analysis.get('all_extensions', [])
                for extension in all_extensions:
                    if 'arm_velocity' in extension:
                        arm_velocities.append(extension['arm_velocity'])
            
            # If no attack_analysis, try to extract from launches and extensions directly
            if not front_foot_velocities and not hip_velocities and not arm_velocities:
                # Try to extract from launches data structure
                launches = data.get('launches', [])
                for launch in launches:
                    if isinstance(launch, dict):
                        if 'front_foot_max_velocity' in launch:
                            front_foot_velocities.append(launch['front_foot_max_velocity'])
                        if 'front_hip_max_velocity' in launch:
                            hip_velocities.append(launch['front_hip_max_velocity'])
                
                # Try to extract from extensions data structure
                extensions = data.get('extensions', [])
                for extension in extensions:
                    if isinstance(extension, dict):
                        if 'max_velocity' in extension:
                            arm_velocities.append(extension['max_velocity'])
                        elif 'arm_velocity' in extension:
                            arm_velocities.append(extension['arm_velocity'])
            
            # Calculate averages
            data['avg_front_foot_velocity'] = np.mean(front_foot_velocities) if front_foot_velocities else 0.0
            data['avg_hip_velocity_launch'] = np.mean(hip_velocities) if hip_velocities else 0.0
            data['avg_arm_velocity_extension'] = np.mean(arm_velocities) if arm_velocities else 0.0
            
            if not front_foot_velocities and not hip_velocities and not arm_velocities:
                logging.warning(f"No velocity data found for {side} fencer in match {bout.get('match_idx', 'unknown')}")
            else:
                logging.info(f"Extracted velocities for {side} fencer: front_foot={data['avg_front_foot_velocity']:.3f}, hip={data['avg_hip_velocity_launch']:.3f}, arm={data['avg_arm_velocity_extension']:.3f}")
    
    return bout_data

def compute_derived_metrics(bout_data, fps=30):
    """Compute additional metrics for each fencer in each bout."""
    # Extract velocity metrics from attack details first
    bout_data = extract_velocity_metrics_from_attacks(bout_data)
    
    for bout in bout_data:
        total_frames = bout['frame_range'][1] - bout['frame_range'][0] + 1
        is_long_bout = total_frames > 80
        
        # Compute first step metrics if not already available
        for side in ['left', 'right']:
            if 'first_step' not in bout[f'{side}_data'] or bout[f'{side}_data']['first_step']['init_time'] == float('inf'):
                left_first_step, right_first_step = compute_first_step_metrics(bout['left_x_df'], bout['right_x_df'], fps)
                bout['left_data']['first_step'] = left_first_step
                bout['right_data']['first_step'] = right_first_step
                break
        
        for side in ['left', 'right']:
            data = bout[f'{side}_data']
            
            # Ensure required data structures exist
            if 'advance' not in data:
                data['advance'] = []
            if 'pause' not in data:
                data['pause'] = []
            if 'arm_extensions' not in data:
                data['arm_extensions'] = []
            
            # Calculate time-based intervals
            data['advance_sec'] = [(start / fps, end / fps) for start, end in data['advance']]
            data['pause_sec'] = [(start / fps, end / fps) for start, end in data['pause']]
            data['arm_extensions_sec'] = [(start / fps, end / fps) for start, end in data['arm_extensions']]
            
            # Calculate ratios - use existing values if available, otherwise compute
            if 'advance_ratio' not in data:
                advance_frames = sum(end - start + 1 for start, end in data['advance'])
                pause_frames = sum(end - start + 1 for start, end in data['pause'])
                total_frames = advance_frames + pause_frames
                data['advance_ratio'] = advance_frames / total_frames if total_frames > 0 else 0
                data['pause_ratio'] = pause_frames / total_frames if total_frames > 0 else 0
            
            # Calculate arm extension metrics - use existing values if available
            if 'arm_extension_freq' not in data:
                data['arm_extension_freq'] = len(data['arm_extensions'])
            if 'avg_arm_extension_duration' not in data:
                data['avg_arm_extension_duration'] = np.mean([end - start for start, end in data['arm_extensions_sec']]) if data['arm_extensions'] else 0
            
            # Calculate launch promptness - use existing value if available
            if 'launch_promptness' not in data:
                if data.get('has_launch', False) and data['arm_extensions']:
                    first_extension_start = min(start for start, end in data['arm_extensions_sec'])
                    launch_time = data.get('launch_frame', -1) / fps if data.get('launch_frame', -1) != -1 else float('inf')
                    data['launch_promptness'] = launch_time - first_extension_start if launch_time != float('inf') else float('inf')
                else:
                    data['launch_promptness'] = float('inf')
            
            if is_long_bout:
                if 'attacking_score' not in data:
                    advance_late = sum(end - start for start, end in data['advance_sec'] if start >= 20/fps)
                    data['attacking_score'] = advance_late
                if 'is_attacking' not in data:
                    data['is_attacking'] = False
            else:
                # For short bouts, calculate timing metrics - use existing values if available
                if 'first_pause_time' not in data or data['first_pause_time'] is None:
                    pause_starts = [start for start, end in data['pause_sec'] if start is not None]
                    data['first_pause_time'] = min(pause_starts) if pause_starts else float('inf')
                
                if 'first_restart_time' not in data or data['first_restart_time'] is None:
                    restart_times = [start for start, end in data['advance_sec'] if start is not None and any((p_end is not None) and (p_end <= start) for p_start, p_end in data['pause_sec'])]
                    data['first_restart_time'] = min(restart_times) if restart_times else float('inf')
                
                if 'post_pause_velocity' not in data:
                    post_pause_velocity = []
                    post_pause_acceleration = []
                    frame_numbers = bout.get('frame_numbers', [])
                    front_foot_x = data.get('front_foot_x', [])
                    
                    if frame_numbers and front_foot_x:
                        for p_start, p_end in data['pause_sec']:
                            post_pause_frames = [f for f in frame_numbers if p_end * fps <= f / fps <= (p_end + 0.5) * fps]
                            if post_pause_frames and len(post_pause_frames) > 1:
                                try:
                                    x = [front_foot_x[frame_numbers.index(f)] for f in post_pause_frames if f in frame_numbers]
                                    if len(x) > 1:
                                        v = np.diff(x) * fps
                                        post_pause_velocity.append(np.mean(v) if len(v) > 0 else 0)
                                        if len(v) > 1:
                                            a = np.diff(v) * fps * fps
                                            post_pause_acceleration.append(np.mean(a) if len(a) > 0 else 0)
                                except (IndexError, ValueError):
                                    continue
                    
                    data['post_pause_velocity'] = np.mean(post_pause_velocity) if post_pause_velocity else data.get('velocity', 0)
                    data['post_pause_acceleration'] = np.mean(post_pause_acceleration) if post_pause_acceleration else data.get('acceleration', 0)
                
                if 'right_of_way_score' not in data:
                    fp = data.get('first_pause_time')
                    fr = data.get('first_restart_time')
                    ppv = data.get('post_pause_velocity', 0) or 0
                    aef = data.get('arm_extension_freq', 0) or 0
                    fp_term = (-fp) if (isinstance(fp, (int, float)) and not math.isinf(fp)) else 0
                    fr_term = (-fr) if (isinstance(fr, (int, float)) and not math.isinf(fr)) else 0
                    data['right_of_way_score'] = fp_term + fr_term + float(ppv) + float(aef)
        
        if is_long_bout:
            left_score = bout['left_data']['attacking_score']
            right_score = bout['right_data']['attacking_score']
            if left_score > right_score:
                bout['left_data']['is_attacking'] = True
            elif right_score > left_score:
                bout['right_data']['is_attacking'] = True
    
    all_velocities = []
    for bout in bout_data:
        left_fs = (bout.get('left_data', {}) or {}).get('first_step', {}) or {}
        right_fs = (bout.get('right_data', {}) or {}).get('first_step', {}) or {}
        lv = left_fs.get('velocity', 0) or 0
        rv = right_fs.get('velocity', 0) or 0
        if isinstance(lv, (int, float)) and lv != 0:
            all_velocities.append(lv)
        if isinstance(rv, (int, float)) and rv != 0:
            all_velocities.append(rv)
    median_velocity = np.median(all_velocities) if all_velocities else 0
    
    for bout in bout_data:
        for side in ['left', 'right']:
            data = bout.get(f'{side}_data', {}) or {}
            fs = data.get('first_step', {}) or {}
            v = fs.get('velocity', 0) or 0
            fs['is_fast'] = (v > median_velocity) if isinstance(v, (int, float)) and v != 0 else False
            # write back in case dicts are separate objects
            data['first_step'] = fs
            bout[f'{side}_data'] = data
    
    # Detect steps for each bout
    for bout in bout_data:
        left_steps = detect_steps_with_pauses(bout['left_x_df'], bout['left_y_df'], "Left Fencer")
        right_steps = detect_steps_with_pauses(bout['right_x_df'], bout['right_y_df'], "Right Fencer")
        bout['left_data']['steps'] = left_steps
        bout['right_data']['steps'] = right_steps
    
    logging.info("Derived metrics and steps computed")
    return bout_data

def detect_steps_with_pauses(x_data, y_data, fencer_name):
    """Detect steps based on pause detection."""
    num_frames = len(x_data)
    steps = []
    static_frames = []
    
    # Assume key points 8 and 9 are the left and right feet
    foot1_idx, foot2_idx = 8, 9  # Adjust if different key points represent feet
    
    # Arrays to store changes
    displacements_x = np.zeros(num_frames - 1)
    distance_changes = np.zeros(num_frames - 1)
    y_changes = np.zeros(num_frames - 1)
    
    # Calculate changes between consecutive frames
    for i in range(num_frames - 1):
        x_current = x_data.iloc[i].values
        x_next = x_data.iloc[i + 1].values
        y_current = y_data.iloc[i].values
        y_next = y_data.iloc[i + 1].values
        
        # Median x-displacement across all key points
        displacements_x[i] = np.median(x_next - x_current)
        
        # Calculate Euclidean distance change between feet
        dist_current = np.sqrt((x_current[foot1_idx] - x_current[foot2_idx])**2 + 
                               (y_current[foot1_idx] - y_current[foot2_idx])**2)
        dist_next = np.sqrt((x_next[foot1_idx] - x_next[foot2_idx])**2 + 
                            (y_next[foot1_idx] - y_next[foot2_idx])**2)
        distance_changes[i] = abs(dist_next - dist_current)  # Absolute change
        
        # Average y-change for feet
        y_changes[i] = np.mean([abs(y_next[foot1_idx] - y_current[foot1_idx]), 
                               abs(y_next[foot2_idx] - y_current[foot2_idx])])
    
    # Smooth changes with a rolling mean
    window_size = 3
    smoothed_displacements_x = pd.Series(displacements_x).rolling(
        window=window_size, min_periods=1, center=True
    ).mean().to_numpy()
    smoothed_distance_changes = pd.Series(distance_changes).rolling(
        window=window_size, min_periods=1, center=True
    ).mean().to_numpy()
    smoothed_y_changes = pd.Series(y_changes).rolling(
        window=window_size, min_periods=1, center=True
    ).mean().to_numpy()
    
    # Detect static (pause) frames
    static_threshold = 0.05  # Threshold for minimal change (adjust as needed)
    for i in range(len(smoothed_displacements_x)):
        if (abs(smoothed_displacements_x[i]) < static_threshold and 
            smoothed_distance_changes[i] < static_threshold and 
            smoothed_y_changes[i] < static_threshold):
            static_frames.append(i + 1)  # Adjust index to match frame after change
    
    # Include the first frame as a static start if it meets criteria
    if num_frames > 1 and (abs(smoothed_displacements_x[0]) < static_threshold and 
                           smoothed_distance_changes[0] < static_threshold and 
                           smoothed_y_changes[0] < static_threshold):
        static_frames.insert(0, 0)
    
    # Include the last frame as a static end if it would be part of a movement
    if num_frames - 1 not in static_frames and num_frames > 1:
        static_frames.append(num_frames - 1)
    
    # Define movement intervals between static frames
    movement_intervals = []
    for i in range(len(static_frames) - 1):
        start_frame = static_frames[i]
        end_frame = static_frames[i + 1]
        if end_frame - start_frame > 1:  # Ensure interval has movement frames
            movement_intervals.append((start_frame, end_frame))
    
    # Analyze each movement interval
    for start_frame, end_frame in movement_intervals:
        interval_displacements_x = displacements_x[start_frame:end_frame - 1]  # Up to second-to-last frame
        interval_distance_changes = distance_changes[start_frame:end_frame - 1]
        interval_y_changes = y_changes[start_frame:end_frame - 1]
        
        # Calculate total changes
        total_displacement_x = np.mean(interval_displacements_x) * (end_frame - start_frame - 1)  # Scaled by frame count
        total_distance_change = np.sum(interval_distance_changes)  # Cumulative change
        total_y_change = np.sum(interval_y_changes)  # Cumulative y movement
        
        # Determine step type
        if fencer_name == "Left Fencer":
            step_type = "Forward" if total_displacement_x > 0 else "Backward"
        else:  # Right Fencer
            step_type = "Forward" if total_displacement_x < 0 else "Backward"
        
        # Calculate step metrics
        magnitude = abs(total_displacement_x) + abs(total_distance_change) + abs(total_y_change)
        size = "Large" if magnitude > 0.7 else "Small"  # Adjusted threshold
        duration = (end_frame - start_frame) * (1 / 30)  # Use fps from argument
        speed = magnitude / duration if duration > 0 else 0
        pace = "Fast" if speed > 2 else "Slow"
        
        steps.append({
            "Start Frame": start_frame,
            "End Frame": end_frame,
            "Type": step_type,
            "Size": size,
            "Speed": pace,
            "Displacement": abs(total_displacement_x),
            "Distance Change": abs(total_distance_change),
            "Y Change": abs(total_y_change),
            "Duration (s)": duration,
            "Speed (units/s)": speed
        })
    
    return steps

def aggregate_fencer_data(bout_data):
    """Aggregate metrics across bouts for each fencer."""
    fencer_data = {
        'Fencer_Left': {
            'bouts': [],
            'first_step_init': [],
            'first_step_velocity': [],
            'first_step_acceleration': [],
            'velocity': [],
            'acceleration': [],
            'advance_ratio': [],
            'pause_ratio': [],
            'arm_extension_freq': [],
            'avg_arm_extension_duration': [],
            'launch_promptness': [],
            'attacking_count': 0,
            'right_of_way_scores': [],
            'total_bouts': 0,
            'front_foot_velocities': [],
            'hip_velocities': [],
            'arm_velocities': []
        },
        'Fencer_Right': {
            'bouts': [],
            'first_step_init': [],
            'first_step_velocity': [],
            'first_step_acceleration': [],
            'velocity': [],
            'acceleration': [],
            'advance_ratio': [],
            'pause_ratio': [],
            'arm_extension_freq': [],
            'avg_arm_extension_duration': [],
            'launch_promptness': [],
            'attacking_count': 0,
            'right_of_way_scores': [],
            'total_bouts': 0,
            'front_foot_velocities': [],
            'hip_velocities': [],
            'arm_velocities': []
        }
    }
    
    for bout in bout_data:
        total_frames = bout['frame_range'][1] - bout['frame_range'][0] + 1
        is_long_bout = total_frames > 80
        
        for side, fencer_id in [('left', 'Fencer_Left'), ('right', 'Fencer_Right')]:
            data = bout[f'{side}_data']
            fencer_data[fencer_id]['bouts'].append({
                'match_idx': bout['match_idx'],
                'metrics': data,
                'is_long_bout': is_long_bout
            })
            fencer_data[fencer_id]['first_step_init'].append(data['first_step']['init_time'])
            fencer_data[fencer_id]['first_step_velocity'].append(data['first_step']['velocity'])
            fencer_data[fencer_id]['first_step_acceleration'].append(data['first_step']['acceleration'])
            fencer_data[fencer_id]['velocity'].append(data['velocity'])
            fencer_data[fencer_id]['acceleration'].append(data['acceleration'])
            fencer_data[fencer_id]['advance_ratio'].append(data['advance_ratio'])
            fencer_data[fencer_id]['pause_ratio'].append(data['pause_ratio'])
            fencer_data[fencer_id]['arm_extension_freq'].append(data['arm_extension_freq'])
            fencer_data[fencer_id]['avg_arm_extension_duration'].append(data['avg_arm_extension_duration'])
            fencer_data[fencer_id]['launch_promptness'].append(data['launch_promptness'] if data['launch_promptness'] != float('inf') else None)
            fencer_data[fencer_id]['front_foot_velocities'].append(data.get('avg_front_foot_velocity', 0.0))
            fencer_data[fencer_id]['hip_velocities'].append(data.get('avg_hip_velocity_launch', 0.0))
            fencer_data[fencer_id]['arm_velocities'].append(data.get('avg_arm_velocity_extension', 0.0))
            if is_long_bout and data.get('is_attacking', False):
                fencer_data[fencer_id]['attacking_count'] += 1
            if not is_long_bout:
                fencer_data[fencer_id]['right_of_way_scores'].append(data['right_of_way_score'])
            fencer_data[fencer_id]['total_bouts'] += 1
    
    for fencer_id, data in fencer_data.items():
        data['avg_first_step_init'] = np.mean([t for t in data['first_step_init'] if t != float('inf')]) if any(t != float('inf') for t in data['first_step_init']) else float('inf')
        data['avg_first_step_velocity'] = np.mean([v for v in data['first_step_velocity'] if v != 0]) if any(v != 0 for v in data['first_step_velocity']) else 0
        data['avg_first_step_acceleration'] = np.mean([a for a in data['first_step_acceleration'] if a != 0]) if any(a != 0 for a in data['first_step_acceleration']) else 0
        data['avg_velocity'] = np.mean(data['velocity'])
        data['std_velocity'] = np.std(data['velocity'])
        data['avg_acceleration'] = np.mean(data['acceleration'])
        data['std_acceleration'] = np.std(data['acceleration'])
        data['avg_advance_ratio'] = np.mean(data['advance_ratio'])
        data['avg_pause_ratio'] = np.mean(data['pause_ratio'])
        data['total_arm_extensions'] = sum(data['arm_extension_freq'])
        data['avg_arm_extension_duration'] = np.mean([d for d in data['avg_arm_extension_duration'] if d > 0]) if any(d > 0 for d in data['avg_arm_extension_duration']) else 0
        valid_promptness = [p for p in data['launch_promptness'] if p is not None]
        data['avg_launch_promptness'] = np.mean(valid_promptness) if valid_promptness else float('inf')
        data['attacking_ratio'] = data['attacking_count'] / sum(1 for bout in data['bouts'] if bout['is_long_bout']) if any(bout['is_long_bout'] for bout in data['bouts']) else 0
        valid_scores = [s for s in data['right_of_way_scores'] if s != 0]
        data['avg_right_of_way_score'] = np.mean(valid_scores) if valid_scores else 0
        # Calculate average velocity metrics
        data['avg_front_foot_velocity'] = np.mean([v for v in data['front_foot_velocities'] if v > 0]) if any(v > 0 for v in data['front_foot_velocities']) else 0
        data['avg_hip_velocity_launch'] = np.mean([v for v in data['hip_velocities'] if v > 0]) if any(v > 0 for v in data['hip_velocities']) else 0
        data['avg_arm_velocity_extension'] = np.mean([v for v in data['arm_velocities'] if v > 0]) if any(v > 0 for v in data['arm_velocities']) else 0
    
    logging.info("Fencer data aggregated")
    return fencer_data

def normalize_metrics_to_radar_scale(fencer_data):
    """Normalize fencer metrics to a 1-5 scale for radar chart where 5 is always best."""
    
    # Define metric ranges and whether higher is better
    # For each metric: (min_expected, max_expected, higher_is_better)
    metric_ranges = {
        'avg_first_step_init': (0.0, 0.2, False),  # Lower start time is better
        'avg_first_step_velocity': (0.0, 2.0, True),  # Higher velocity is better
        'avg_first_step_acceleration': (-2.0, 3.0, True),  # Higher acceleration is better
        'avg_velocity': (0.0, 0.15, True),  # Higher velocity is better
        'std_velocity': (0.0, 0.05, False),  # Lower variance is better (more stable)
        'avg_acceleration': (-0.01, 0.01, True),  # Higher acceleration is better
        'avg_advance_ratio': (0.0, 1.0, True),  # Higher advance ratio is better
        'avg_pause_ratio': (0.0, 1.0, False),  # Lower pause ratio is better
        'total_arm_extensions': (0, 20, True),  # More extensions is better
        'avg_arm_extension_duration': (0.0, 1.0, True),  # Longer duration can be better
        'attacking_ratio': (0.0, 1.0, True),  # Higher attack ratio is better
        'avg_front_foot_velocity': (0.0, 0.15, True),  # Higher velocity is better
        'avg_hip_velocity_launch': (0.0, 0.12, True),  # Higher velocity is better
        'avg_arm_velocity_extension': (0.0, 0.08, True),  # Higher velocity is better
    }
    
    def normalize_value(value, min_val, max_val, higher_is_better):
        """Normalize a single value to 1-5 scale."""
        if value is None or value == float('inf') or value == float('-inf'):
            return 1.0
        
        # Clamp value to expected range
        clamped = max(min_val, min(max_val, value))
        
        # Normalize to 0-1
        if max_val == min_val:
            normalized = 0.5
        else:
            normalized = (clamped - min_val) / (max_val - min_val)
        
        # If lower is better, invert the scale
        if not higher_is_better:
            normalized = 1.0 - normalized
        
        # Scale to 1-5 range
        return 1.0 + (normalized * 4.0)
    
    normalized_data = {}
    
    for fencer_id, data in fencer_data.items():
        normalized_data[fencer_id] = {}
        
        for metric, (min_val, max_val, higher_is_better) in metric_ranges.items():
            raw_value = data.get(metric, 0)
            normalized_data[fencer_id][metric] = normalize_value(raw_value, min_val, max_val, higher_is_better)
    
    return normalized_data

import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties

# Set Chinese font (prefer Simplified Chinese)
plt.rcParams['font.sans-serif'] = [
    'Noto Sans CJK SC',
    'WenQuanYi Micro Hei',
    'WenQuanYi Zen Hei',
    'Noto Sans SC',
    'SimHei',
    'Microsoft YaHei',
    'DejaVu Sans',
    'sans-serif'
]
plt.rcParams['axes.unicode_minus'] = False

# Metric Translations
METRIC_TRANSLATIONS = {
    'Initial Step Time (s)': 'Initial Step Time (s)',
    'Initial Step Velocity': 'Initial Step Velocity',
    'Initial Step Acceleration': 'Initial Step Acceleration',
    'Average Velocity': 'Average Velocity',
    'Velocity Stability': 'Velocity Stability',
    'Average Acceleration': 'Average Acceleration',
    'Advance Ratio': 'Advance Ratio',
    'Pause Ratio': 'Pause Ratio',
    'Total Arm Extensions': 'Total Arm Extensions',
    'Avg. Extension Duration': 'Avg. Extension Duration',
    'Attack Frequency': 'Attack Frequency',
    'Front Foot Velocity': 'Front Foot Velocity',
    'Lunge Hip Velocity': 'Lunge Hip Velocity',
    'Arm Extension Velocity': 'Arm Extension Velocity',
    # For radar chart specifically
    'Initial Step Time': 'Initial Step Time',
    'Avg Velocity': 'Avg Velocity',
    'Arm Extensions': 'Arm Extensions',
    'Extension Duration': 'Extension Duration',
}

def generate_radar_chart(fencer_data, output_dir):
    """Generate a radar chart comparing fencers across normalized metrics."""
    
    import numpy as np
    import os
    import logging
    import traceback
    
    try:
        # Normalize metrics to 1-5 scale
        normalized_data = normalize_metrics_to_radar_scale(fencer_data)
        
        # Define metrics for radar chart
        radar_metrics = {
            '首步时间': 'avg_first_step_init',
            '首步速度': 'avg_first_step_velocity', 
            '首步加速度': 'avg_first_step_acceleration',
            '平均速度': 'avg_velocity',
            '速度稳定性': 'std_velocity',
            '平均加速度': 'avg_acceleration',
            '前进比率': 'avg_advance_ratio',
            '停顿比率': 'avg_pause_ratio',
            '手臂伸展': 'total_arm_extensions',
            '伸展持续时间': 'avg_arm_extension_duration',
            '攻击频率': 'attacking_ratio',
            '前脚速度': 'avg_front_foot_velocity',
            '弓步髋部速度': 'avg_hip_velocity_launch',
            '手臂伸展速度': 'avg_arm_velocity_extension'
        }
        
        # Extract data for radar chart
        left_data = normalized_data.get('Fencer_Left', {})
        right_data = normalized_data.get('Fencer_Right', {})
        
        categories = []
        left_values = []
        right_values = []
        
        for display_name, key in radar_metrics.items():
            left_val = left_data.get(key, 1.0)
            right_val = right_data.get(key, 1.0)
            
            # Only include metrics where at least one value is meaningful
            if left_val > 1.0 or right_val > 1.0:
                categories.append(METRIC_TRANSLATIONS.get(display_name, display_name))
                left_values.append(left_val)
                right_values.append(right_val)
        
        if not categories:
            logging.warning("No valid metrics found for radar chart")
            return None
        
        # Number of variables
        N = len(categories)
        
        # Compute angle for each axis
        angles = [n / float(N) * 2 * np.pi for n in range(N)]
        angles += angles[:1]  # Complete the circle
        
        # Close the plot
        left_values += left_values[:1]
        right_values += right_values[:1]
        
        # Create radar chart
        fig, ax = plt.subplots(figsize=(12, 12), subplot_kw=dict(projection='polar'))
        
        # Plot data
        ax.plot(angles, left_values, 'o-', linewidth=2, label='左侧击剑手', color='#1f77b4')
        ax.fill(angles, left_values, alpha=0.25, color='#1f77b4')
        
        ax.plot(angles, right_values, 'o-', linewidth=2, label='右侧击剑手', color='#ff7f0e')
        ax.fill(angles, right_values, alpha=0.25, color='#ff7f0e')
        
        # Add category labels
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(categories, fontsize=10)
        
        # Set y-axis
        ax.set_ylim(0, 5)
        ax.set_yticks([1, 2, 3, 4, 5])
        ax.set_yticklabels(['1', '2', '3', '4', '5'])
        ax.grid(True)
        
        # Add title and legend
        plt.title('击剑手性能雷达图\n(1=弱, 5=优秀)', size=16, fontweight='bold', pad=20)
        
        plt.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))
        
        # Save chart
        plt.tight_layout()
        radar_path = os.path.join(output_dir, 'fencer_radar_comparison.png')
        plt.savefig(radar_path, dpi=300, bbox_inches='tight', 
                   facecolor='white', edgecolor='none', format='png')
        plt.close()
        
        logging.info(f"Successfully saved radar chart: {radar_path}")
        return radar_path
        
    except Exception as e:
        logging.error(f"Error generating radar chart: {str(e)}\n{traceback.format_exc()}")
        return None

def generate_fencer_comparison_chart(fencer_data, output_dir):
    """Generate a comparison bar chart between left and right fencers."""
    
    import matplotlib.pyplot as plt
    import numpy as np
    import os
    import logging
    import traceback
    
    try:
        # Define Chinese metrics
        metrics = {
            '首步时间 (秒)': 'avg_first_step_init',
            '首步速度': 'avg_first_step_velocity', 
            '首步加速度': 'avg_first_step_acceleration',
            '平均速度': 'avg_velocity',
            '速度稳定性': 'std_velocity',
            '平均加速度': 'avg_acceleration',
            '前进比率': 'avg_advance_ratio',
            '停顿比率': 'avg_pause_ratio',
            '手臂伸展总数': 'total_arm_extensions',
            '平均伸展持续时间': 'avg_arm_extension_duration',
            '攻击频率': 'attacking_ratio',
            '前脚速度': 'avg_front_foot_velocity',
            '弓步髋部速度': 'avg_hip_velocity_launch',
            '手臂伸展速度': 'avg_arm_velocity_extension'
        }
        
        chart_title = '击剑手性能对比'
        x_label = '指标'
        y_label = '数值'
        left_label = '左侧击剑手'
        right_label = '右侧击剑手'
        
        # Process data
        left_data = fencer_data.get('Fencer_Left', {})
        right_data = fencer_data.get('Fencer_Right', {})
        
        left_values = []
        right_values = []
        metric_names = []
        
        for display_name, key in metrics.items():
            left_val = left_data.get(key, 0)
            right_val = right_data.get(key, 0)
            
            if key == 'avg_launch_promptness':
                if left_val == float('inf'): left_val = 0
                if right_val == float('inf'): right_val = 0
            
            if key == 'std_velocity':
                left_val *= 100
                right_val *= 100
            
            if left_val != 0 or right_val != 0:
                left_values.append(left_val)
                right_values.append(right_val)
                metric_names.append(METRIC_TRANSLATIONS.get(display_name, display_name))
        
        if not metric_names:
            logging.warning("No valid metrics found for comparison chart")
            return None
        
        # Create chart
        fig, ax = plt.subplots(figsize=(16, 10))
        
        x = np.arange(len(metric_names))
        width = 0.35
        
        bars1 = ax.bar(x - width/2, left_values, width, label=left_label, color='#1f77b4', alpha=0.8)
        bars2 = ax.bar(x + width/2, right_values, width, label=right_label, color='#ff7f0e', alpha=0.8)
        
        # Title, labels
        ax.set_xlabel(x_label, fontsize=12, fontweight='bold')
        ax.set_ylabel(y_label, fontsize=12, fontweight='bold')
        ax.set_title(chart_title, fontsize=16, fontweight='bold', pad=20)
        
        # X and Y tick labels
        ax.set_xticks(x)
        ax.set_xticklabels(metric_names, rotation=45, ha='right')
        
        # Legend
        ax.legend(fontsize=11)
        
        # Grid
        ax.grid(True, alpha=0.3, axis='y')
        
        # Value labels on bars
        def add_value_labels(bars):
            for bar in bars:
                height = bar.get_height()
                if height != 0:
                    ax.annotate(f'{height:.2f}',
                               xy=(bar.get_x() + bar.get_width() / 2, height),
                               xytext=(0, 3),
                               textcoords="offset points",
                               ha='center', va='bottom',
                               fontsize=9)
        
        add_value_labels(bars1)
        add_value_labels(bars2)
        
        # Save chart
        plt.tight_layout()
        chart_path = os.path.join(output_dir, 'fencer_comparison.png')
        
        plt.savefig(chart_path, dpi=300, bbox_inches='tight', facecolor='white', edgecolor='none', format='png')
        plt.close()
        
        logging.info(f"Successfully saved comparison chart: {chart_path}")
        return chart_path
        
    except Exception as e:
        logging.error(f"Error generating comparison chart: {str(e)}\n{traceback.format_exc()}")
        return None



def generate_chart_explanation(fencer_data):
    """Generate GPT explanation for the fencer comparison chart."""
    try:
        # Prepare data summary for GPT
        left_data = fencer_data.get('Fencer_Left', {})
        right_data = fencer_data.get('Fencer_Right', {})
        
        prompt = f"""
        您是一位专业的佩剑击剑分析师。我为两名击剑手（左侧击剑手vs右侧击剑手）创建了两个对比图表来显示性能指标：一个显示原始值的柱状图和一个标准化分数从1-5（5为最优）的雷达图。请提供详细的中文分析，解释这些图表揭示的每位击剑手的优势和劣势。

        **左侧击剑手指标：**
        - 首步时间: {left_data.get('avg_first_step_init', 'N/A'):.3f}秒
        - 首步速度: {left_data.get('avg_first_step_velocity', 0):.3f}
        - 首步加速度: {left_data.get('avg_first_step_acceleration', 0):.3f}
        - 平均速度: {left_data.get('avg_velocity', 0):.3f}
        - 速度稳定性: {left_data.get('std_velocity', 0)*100:.3f}
        - 平均加速度: {left_data.get('avg_acceleration', 0):.3f}
        - 前进比率: {left_data.get('avg_advance_ratio', 0):.3f}
        - 暂停比率: {left_data.get('avg_pause_ratio', 0):.3f}
        - 总伸臂次数: {left_data.get('total_arm_extensions', 0)}
        - 平均伸臂持续时间: {left_data.get('avg_arm_extension_duration', 0):.3f}秒
        - 攻击频率: {left_data.get('attacking_ratio', 0):.3f}
        - 前脚速度: {left_data.get('avg_front_foot_velocity', 0):.3f}
        - 冲刺臀部速度: {left_data.get('avg_hip_velocity_launch', 0):.3f}
        - 伸臂速度: {left_data.get('avg_arm_velocity_extension', 0):.3f}

        **右侧击剑手指标：**
        - 首步时间: {right_data.get('avg_first_step_init', 'N/A'):.3f}秒
        - 首步速度: {right_data.get('avg_first_step_velocity', 0):.3f}
        - 首步加速度: {right_data.get('avg_first_step_acceleration', 0):.3f}
        - 平均速度: {right_data.get('avg_velocity', 0):.3f}
        - 速度稳定性: {right_data.get('std_velocity', 0)*100:.3f}
        - 平均加速度: {right_data.get('avg_acceleration', 0):.3f}
        - 前进比率: {right_data.get('avg_advance_ratio', 0):.3f}
        - 暂停比率: {right_data.get('avg_pause_ratio', 0):.3f}
        - 总伸臂次数: {right_data.get('total_arm_extensions', 0)}
        - 平均伸臂持续时间: {right_data.get('avg_arm_extension_duration', 0):.3f}秒
        - 攻击频率: {right_data.get('attacking_ratio', 0):.3f}
        - 前脚速度: {right_data.get('avg_front_foot_velocity', 0):.3f}
        - 冲刺臀部速度: {right_data.get('avg_hip_velocity_launch', 0):.3f}
        - 伸臂速度: {right_data.get('avg_arm_velocity_extension', 0):.3f}

        请提供包含以下内容的综合分析：
        1.  **整体性能总结**：哪位击剑手显得更强，为什么？
        2.  **具体优势**：基于指标，每位击剑手的关键优势是什么？
        3.  **需要改进的领域**：数据揭示了什么弱点？
        4.  **战术洞察**：这些指标对每位击剑手的风格和策略有什么提示？
        5.  **训练建议**：基于对比分析提供具体的改进建议。

        使用专业击剑术语，提供对教练和运动员有价值的可操作见解。分析应该详细、专业且易于理解。所有回复请用中文。
        """

        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "您是一位专业的佩剑击剑分析师，提供详细的中文性能对比。"},
                {"role": "user", "content": prompt}
            ],
            temperature=0.7,
            max_tokens=3000
        )
        
        explanation = response.choices[0].message.content
        logging.info("Generated chart explanation via GPT")
        return explanation
        
    except Exception as e:
        logging.error(f"Error generating chart explanation: {str(e)}\n{traceback.format_exc()}")
        return f"Chart analysis is temporarily unavailable due to an API error: {str(e)}"

def convert_numpy_types(obj):
    if isinstance(obj, np.bool_):
        return bool(obj)
    elif isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        # Handle special float values
        if np.isnan(obj):
            return None
        elif np.isinf(obj):
            return None
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, pd.DataFrame):
        return f"DataFrame with shape {obj.shape}"
    elif isinstance(obj, dict):
        return {k: convert_numpy_types(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_types(item) for item in obj]
    elif isinstance(obj, tuple):
        return tuple(convert_numpy_types(item) for item in obj)
    # Handle special Python float values
    elif isinstance(obj, float):
        if np.isnan(obj) or np.isinf(obj):
            return None
        return obj
    return obj

def create_bout_data_for_gpt(bout):
    """Create a sanitized version of bout data for GPT analysis, excluding DataFrames."""
    sanitized_bout = {}
    
    # Copy all fields except DataFrames
    for key, value in bout.items():
        if key in ['left_x_df', 'left_y_df', 'right_x_df', 'right_y_df']:
            # Skip DataFrame data - it's not needed for GPT analysis
            continue
        else:
            sanitized_bout[key] = convert_numpy_types(value)
    
    return sanitized_bout

def generate_plots(fencer_data, output_dir):
    """Generate visualizations for fencer metrics."""
    logging.info(f"Generating plots in {output_dir}")
    os.makedirs(output_dir, exist_ok=True)
    
    if not fencer_data or not any(data['bouts'] for data in fencer_data.values()):
        logging.error("No bout data available for plotting")
        return None
    
    # Generate existing plots
    try:
        plt.figure(figsize=(10, 6))
        for fencer_id, data in fencer_data.items():
            init_times = [bout['metrics']['first_step']['init_time'] for bout in data['bouts']]
            match_indices = [bout['match_idx'] for bout in data['bouts']]
            label = 'Left Fencer' if fencer_id == 'Fencer_Left' else 'Right Fencer'
            plt.plot(match_indices, init_times, marker='o', label=label)
        plt.title('First Step Lunge Time per Bout')
        plt.xlabel('Bout Index')
        plt.ylabel('Lunge Time (s)')
        plt.legend()
        plt.grid(True)
        plot_path = os.path.join(output_dir, 'first_step_init.png')
        plt.savefig(plot_path)
        plt.close()
        logging.info(f"Saved plot: {plot_path}")
    except Exception as e:
        logging.error(f"Error generating first_step_init plot: {str(e)}\n{traceback.format_exc()}")
    
    try:
        plt.figure(figsize=(10, 6))
        for fencer_id, data in fencer_data.items():
            velocities = [bout['metrics']['velocity'] for bout in data['bouts']]
            match_indices = [bout['match_idx'] for bout in data['bouts']]
            label = 'Left Fencer' if fencer_id == 'Fencer_Left' else 'Right Fencer'
            plt.plot(match_indices, velocities, marker='o', label=label)
        plt.title('Velocity per Bout')
        plt.xlabel('Bout Index')
        plt.ylabel('Average Velocity')
        plt.legend()
        plt.grid(True)
        plot_path = os.path.join(output_dir, 'velocity_trend.png')
        plt.savefig(plot_path)
        plt.close()
        logging.info(f"Saved plot: {plot_path}")
    except Exception as e:
        logging.error(f"Error generating velocity_trend plot: {str(e)}\n{traceback.format_exc()}")

    # Generate new comparison chart and radar chart
    chart_path = generate_fencer_comparison_chart(fencer_data, output_dir)
    radar_path = generate_radar_chart(fencer_data, output_dir)
    
    # Generate graph-specific analysis for radar and comparison charts
    chart_analysis = {}
    
    try:
        from .graph_analysis import generate_graph_analysis
        
        left_metrics = fencer_data.get('Fencer_Left', {})
        right_metrics = fencer_data.get('Fencer_Right', {})
        
        # Generate radar chart analysis
        if radar_path:
            radar_analysis = generate_graph_analysis('radar_chart', left_metrics, right_metrics)
            chart_analysis['radar_chart'] = radar_analysis
            logging.info("Generated radar chart analysis")
        
        # Generate comparison chart analysis
        if chart_path:
            comparison_analysis = generate_graph_analysis('comparison_chart', left_metrics, right_metrics)
            chart_analysis['comparison_chart'] = comparison_analysis
            logging.info("Generated comparison chart analysis")
        
        # Save chart analysis to JSON
        if chart_analysis:
            chart_analysis_file = os.path.join(output_dir, 'chart_analysis.json')
            with open(chart_analysis_file, 'w', encoding='utf-8') as f:
                json.dump(chart_analysis, f, ensure_ascii=False, indent=2)
            logging.info(f"Saved chart analysis to {chart_analysis_file}")
            
    except Exception as e:
        logging.error(f"Error generating chart analysis: {e}")
        chart_analysis = {}
    
    return chart_analysis

def generate_gpt_prompt(bout_data, fencer_data, fps=30):
    """Generate prompt for full analysis report."""
    prompt = """
    您是一位专精佩剑的AI击剑分析师，负责评估多个回合以提供全面的、数据驱动的击剑手性能、策略和战术决策分析。假设每个回合中两名击剑手都得分（没有单灯情况），且挡击反击无法检测。不要明确宣布回合获胜者，因为由于潜在的挡击或单灯情况结果不可靠。相反，基于提供的指标推断哪位击剑手在进攻或优先权方面占优势，专注于定性洞察而非分配数值分数。运用您的竞技佩剑知识提供适合教练和运动员的专业、详细分析，强调击剑手意图、具体动作、其有效性以及跨回合的战略适应。包括步法数据（如前进/后退步伐、速度、持续时间）来评估步法和战术移动。使用自然的中文击剑术语（如"进攻"、"反击"、"转换"、"佯攻"、"步法"、"后退"）和高级佩剑术语（如"复合攻击"、"同时攻击"、"优先权"、"距离管理"、"步法"、"节奏控制"），确保清晰并避免过于细致的、基于测量的语言（如避免"以0.75速度在0.03秒内冲刺"）。将动作呈现为动态的来回交锋以反映佩剑回合的动态性质。分析应该像教练提供可操作反馈一样对话式且专业，有具体例子和明确建议。所有输出应为中文。

    **Input Data**:
    - **Bouts**: {num_bouts} bouts, each containing metrics for left and right fencers.
    - **Fencer IDs**: Left Fencer, Right Fencer
    - **Per-Bout Metrics**:
      - First Step Initiation Time (s): Time to start the first forward step (keypoint 16 moves).
      - First Step Velocity/Acceleration: Average velocity and acceleration over 8 frames after the first step (categorized as fast if above the bout's median velocity).
      - Advance Intervals (s): Time periods of forward movement (Left: x increases, Right: x decreases).
      - Pause/Retreat Intervals (s): Time periods of pause or backward movement.
      - Arm Extension Intervals (s): Intervals indicating potential start of an attack.
      - Has Lunge: Boolean for whether a lunge occurred in the final 15 frames.
      - Lunge Frame: Frame number of the lunge (-1 if none).
      - Velocity/Acceleration: Average velocity and acceleration during the active attack phase (up to the meeting point or lunge peak).
      - Latest Pause/Retreat End (s): End time of the most recent pause/retreat interval.
      - Advance/Pause Ratio: Proportion of frames spent advancing or pausing.
      - Arm Extension Frequency/Duration: Number of arm extensions and their average duration.
      - Lunge Promptness (s): Time from the first arm extension to the lunge (infinity if no lunge).
      - Footwork: Movement intervals with type (Forward/Backward), size (Large/Small), speed (Fast/Slow), displacement, distance change, y-change, duration, and speed (units/s).
      - Bout Result: Left Fencer Wins, Right Fencer Wins, or Unspecified.
      - For Attack-Defense Dynamics (>80 frames):
        - Is Attacking: The fencer who is more advanced after the initial phase (frames 20-30).
      - For In-Fighting (≤80 frames):
        - First Pause Time (s): Earliest pause start time.
        - First Restart Time (s): Earliest advance time after a pause.
        - Post-Pause Velocity: Average velocity after a pause interval.
    - **Per-Fencer Aggregated Metrics**:
      - Average first step initiation time, velocity, acceleration.
      - Average velocity, acceleration (with standard deviation).
      - Average advance/pause ratio.
      - Total arm extensions, average duration.
      - Average lunge promptness.
      - Attack frequency (>80 frames: proportion of bouts in an attacking state).
      - Average footwork duration, speed, and frequency.
    - **Bout Metadata**:
      - Bout Index, Frame Range, FPS: {fps}

    Please use the following terms throughout your response to refer to different actions and intentions.
    - **Compound Attack**: An attack with multiple feints or blade actions (e.g., feint-cut, feint-disengage) to deceive the opponent.
    - **Simultaneous Attack**: Both fencers attack at the same time, with priority determining the score.
    - **Priority**: A rule that gives the right-of-way to the fencer who correctly executes an attack, inferred from arm extension timing and attack continuity.
    - **Distance Management**: Controlling the lunge, retreat, preparation, or breaking distance to induce an opponent or close distance unexpectedly.
    - **Footwork**: Includes advance-lunge (a quick step and lunge) and flèche (a running attack, restricted in modern rules).
    - **Breaking Distance**: Maintaining a distance just outside the opponent's reach to provoke a premature attack, then countering.
    - **Feint**: A deceptive action or blade movement, such as a disengage (slipping under the opponent's parry) or a cut-over (whipping the blade over the opponent's), to bypass their defense.
    - **First and Second Intention**: First intention is a direct attack to score; second intention is to induce a reaction (e.g., a parry, a counter-attack) to set up a subsequent action.
    - **Tempo Control**: Varying the speed of an attack or footwork to disrupt the opponent's rhythm, e.g., slowing the preparation then accelerating into a lunge.

    **Output Format (English)**:
    1. **Fencer Performance Summary**:
       - **Performance Metrics**: Summarize each fencer's aggregated metrics as context, including footwork data:
         - Average first step initiation time, velocity, and acceleration.
         - Average velocity and acceleration (with standard deviation).
         - Average advance/pause ratio.
         - Total arm extensions and average duration.
         - Average lunge promptness.
         - Attack frequency.
         - Average footwork duration, speed, and frequency.
       - **Strategic Tendencies**: Describe each fencer's overall strategy using fencing terminology, including advanced terms and footwork insights:
         - Classify tendencies (e.g., aggressive attacker, defensive counter-attacker, frequent transitions, compound attacks, or second intention).
         - Identify patterns (e.g., early simultaneous attacks for aggression, pauses for breaking distance, fast advance-lunge for dominance, footwork speed for tempo control).
         - Align with saber strategy (e.g., marching attack, counter-time, distance play, priority contests).
       - **Actions & Intent**: Analyze actions and their tactical purpose overall, using advanced terms and footwork data:
         - Discuss how attacks, feints, retreats, and footwork contribute to offensive or priority advantage.
         - Evaluate intent (e.g., compound attack to induce, retreat for breaking distance, second intention via feint, footwork patterns for setup).
         - Assess effectiveness (e.g., did the disengage disrupt the opponent? Did footwork speed enhance tempo control?).
         - Cite specific bouts (e.g., "In Bout 1, the left fencer used a compound attack and fast footwork to force the right fencer to retreat").
       - **Adaptability & Trends**: Examine changes across bouts:
         - Identify changes (e.g., faster advance-lunge, increased feints, consistent footwork via step frequency).
         - Discuss adaptations (e.g., shifting from simultaneous attacks to counter-attacks against an aggressive opponent).
         - Analyze trends (e.g., improved distance management via step size, fatigue impacting tempo control), referencing saber dynamics.
         - Provide specific bout examples.
    When analyzing the above, it's important to look for patterns across bouts: what do they like to do, which actions or habits appear in multiple bouts? Footwork data is important.

    2. **Individual Fencer Analysis**:
       - **Performance Metrics**: Summarize each fencer's aggregated metrics, including footwork data.
       - **Strengths**: Highlight what each fencer does well, using fencing terminology including advanced terms and footwork insights:
         - Identify effective actions (e.g., well-timed compound attack, strong feints like disengage, solid distance control, agile footwork via steps).
         - Provide specific bout examples (e.g., "In Bout 1, the left fencer's quick advance-lunge and footwork control disrupted the right fencer's breaking distance").
         - Ground in saber principles (e.g., priority control, tempo manipulation, footwork precision).
       - **Areas for Improvement**: Identify weaknesses with specific bout examples, including footwork data:
         - Pinpoint issues (e.g., predictable feints in compound attack, slow alternating steps, unstable retreat on flèche, inconsistent footwork speed).
         - Cite bouts (e.g., "In Bout 2, the right fencer's frequent transitions and imbalanced footwork led to a loss of priority").
         - Explain impact by referencing saber principles (e.g., "Overly large steps can lead to a loss of tempo control").
       - **Contextual Recommendations**: Provide actionable advice for similar situations, including footwork adjustments:
         - For effective actions, suggest enhancements (e.g., "The left fencer's compound attack and fast footwork are strong; adding more disengages could increase unpredictability").
         - For weaknesses, suggest corrections (e.g., "The right fencer should quicken their alternating step tempo to counter the opponent's deep attacks").
         - Specify appropriate actions (e.g., "Against a second-intention opponent, initiate with a compound attack and maintain your advance with solid footwork").
         - State whether an action was effective and recommend the ideal action (e.g., "The right fencer's transition step led to being out of position; a small, controlled retreat followed by a quick advance-lunge would have been better").
         - Align with saber strategy (e.g., optimize distance management, control tempo via footwork).
       - **Opponent-Specific Insights**: Analyze opponent tendencies and suggest countermeasures, including footwork data:
         - Identify patterns (e.g., "The opponent favors second intention or frequent simultaneous attacks with quick footwork").
         - Recommend strategies (e.g., "To counter the opponent's breaking distance, use a fast advance-lunge to break priority").
         - Illustrate with a bout example (e.g., "In Bout 3, the opponent's compound attack and footwork control exposed the left fencer's weakness in tempo-control transitions").
    In this section, you should focus on identifying patterns and things they do consistently, judging whether it is correct or not. It needs to be very specific and thorough. Footwork data is important, especially paying attention to whether the fencer has a habit in the initial state of each bout, footwork patterns like continuous small steps or big fast steps, etc.

    3. **Summary Table**:
       - **Comparative Metrics**:
         - First step initiation time, velocity, acceleration.
         - Overall velocity, acceleration.
         - Advance/pause ratio.
         - Arm extensions (frequency, duration).
         - Lunge promptness, attack frequency.
         - Average footwork duration, speed, frequency.
       - **Qualitative Insights**: Explain the importance in saber (e.g., "A fast compound attack and footwork is often a key factor for seizing priority").
       - **Advantaged Fencer**: Indicate the advantage for each metric (e.g., "Left fencer's quick advance-lunge and footwork shows stronger distance management").
       - **Overall Assessment**: Summarize each fencer's strategic profile, key strengths, and critical areas for improvement, using advanced terms and footwork insights.

    4. **Individual Bout Analysis**:
       - **Background**: State you are an AI Saber Analyst, both fencers score, parry-riposte is not detectable, judgment is data-based, and results are for human analyst use.
       - **Action Description**: Narrate the action as an interactive exchange using fencing terminology, including advanced terms and footwork data:
         - Describe the back-and-forth (e.g., "The left fencer was aggressive with a compound attack and fast footwork, while the right fencer retreated and tried to maintain breaking distance").
         - Include key moments (e.g., simultaneous attack, feints like disengage, advance-lunge, retreats, changes in footwork) with rough timings (e.g., "Around 2s, the left fencer initiated a compound attack with fast footwork").
         - Highlight rhythm and intent (e.g., "The right fencer's frequent transition steps and tempo control created, but the footwork was too large, leading to being out of position").
       - **Tactical Analysis**: Analyze intent and effectiveness, including footwork data:
         - Specify the action (e.g., compound attack, simultaneous attack, second intention, footwork pattern).
         - Explain the choice (e.g., "The left fencer used a feint with a disengage and quick footwork to draw a premature attack").
         - Evaluate the outcome (e.g., "The compound attack and footwork effectively induced the opponent, providing a priority opportunity").
         - Use saber principles to highlight decisions/errors (e.g., "The right fencer's unstable retreat and transition steps failed to manage distance effectively").
       - **Priority Inference**: Qualitatively infer the advantage, including footwork data:
         - Based on attack timing, continuity, actions, and footwork speed (e.g., "The left fencer's continuous compound attack and fast footwork likely gave them priority, confidence 70%").
         - Use natural language, not numerical scores.
       - **Performance Evaluation**: Assess each fencer's performance in the bout, including footwork data:
         - Strengths (e.g., "The left fencer's tempo control and footwork were excellent, forcing the opponent to retreat").
         - Weaknesses (e.g., "The right fencer's breaking distance transition and imbalanced footwork failed to close distance effectively").
         - Use specific examples (e.g., "At 2.5s, the left fencer's second intention and footwork positioning exposed the opponent's weakness").
       - **Bout Result Analysis** (if provided):
         - If there's a clear result, analyze how the winning fencer's tactics contributed (e.g., "The left fencer's fast footwork and compound attack likely exploited the right fencer's slow tempo").
         - Analyze the losing fencer's tactical shortcomings (e.g., "The right fencer's frequent transition steps failed to counter the left fencer's attacking rhythm").
         - Remain objective, avoiding referee judgments.
       - **Contextual Recommendations**: Provide actionable advice for similar situations, including footwork adjustments:
         - For effective actions, suggest enhancements (e.g., "The left fencer's compound attack and footwork could incorporate a flèche to add deception").
         - For weaknesses, suggest corrections (e.g., "The right fencer should stabilize their footwork to induce the opponent into a breaking distance").
         - Specify appropriate actions (e.g., "Against a deep-attacking opponent, use tempo control transition steps to force the opponent's hesitation").
         - State whether an action was effective (e.g., "The right fencer's transition step led to being out of position; a small controlled retreat and a quick advance-lunge would have been better").
         - Align with saber strategy (e.g., distance management, tempo control via footwork).

    **Guidelines**:
    - Use natural English fencing terminology, including advanced terms (e.g., compound attack, simultaneous attack, priority, distance management, footwork, tempo control) for clarity.
    - Focus on overall tactical judgment, avoiding excessive metric details in the description.
    - Present actions as a dynamic, back-and-forth exchange, emphasizing rhythm, intent, and footwork patterns.
    - Support observations with data, citing specific bouts (e.g., "In Bout 3, the right fencer's transition step at 3.2s induced a compound attack from the opponent").
    - Reference saber principles (e.g., distance management, priority, tempo control, footwork precision).
    - Provide specific, actionable advice, stating what should be done in similar situations and the ideal action, including footwork adjustments.
    - Ensure the analysis is professional, conversational, and flows logically, connecting metrics and footwork data to tactics.
    - The output should be in English, suitable for fencers and coaches, with a tone of detailed coach feedback.
    - If a bout result is available, analyze how tactics influenced it but remain objective and avoid subjective judgments.
    """
    bout_sections = []
    for bout in bout_data:
        match_idx = bout['match_idx']
        total_frames = bout['frame_range'][1] - bout['frame_range'][0] + 1
        is_long_bout = total_frames > 80

        upload_id = bout['upload_id']
        bout_result_obj = Bout.query.filter_by(upload_id=upload_id, match_idx=match_idx).first()
        result_text = bout_result_obj.result if bout_result_obj else "Unspecified"
        if result_text == 'skip':
            result_text = "Skipped"
        elif result_text == 'left':
            result_text = "Left Wins"
        elif result_text == 'right':
            result_text = "Right Wins"

        left_metrics = ""
        right_metrics = ""
        if is_long_bout:
            left_metrics = f"- Attacking State: {'Attacking' if bout['left_data']['is_attacking'] else 'Defending'}"
            right_metrics = f"- Attacking State: {'Attacking' if bout['right_data']['is_attacking'] else 'Defending'}"
        else:
            left_metrics = (
                f"- First Pause Time: {bout['left_data']['first_pause_time']:.2f} s\n"
                f"  - First Restart Time: {bout['left_data']['first_restart_time']:.2f} s\n"
                f"  - Post-Pause Velocity: {bout['left_data']['post_pause_velocity']:.2f}"
            )
            right_metrics = (
                f"- First Pause Time: {bout['right_data']['first_pause_time']:.2f} s\n"
                f"  - First Restart Time: {bout['right_data']['first_restart_time']:.2f} s\n"
                f"  - Post-Pause Velocity: {bout['right_data']['post_pause_velocity']:.2f}"
            )

        left_pause_end = f"{bout['left_data']['latest_pause_retreat_end'] / fps:.2f}" if bout['left_data']['latest_pause_retreat_end'] != -1 else 'N/A'
        right_pause_end = f"{bout['right_data']['latest_pause_retreat_end'] / fps:.2f}" if bout['right_data']['latest_pause_retreat_end'] != -1 else 'N/A'
        left_launch_promptness = f"{bout['left_data']['launch_promptness']:.2f}" if bout['left_data']['launch_promptness'] != float('inf') else 'N/A'
        right_launch_promptness = f"{bout['right_data']['launch_promptness']:.2f}" if bout['right_data']['launch_promptness'] != float('inf') else 'N/A'

        left_steps = bout['left_data'].get('steps', [])
        right_steps = bout['right_data'].get('steps', [])

        bout_section = f"""
**Bout {match_idx}** ({'Attack-Defense' if is_long_bout else 'In-Fighting'}, {total_frames} frames):
- Frame Range: {bout['frame_range'][0] / fps:.2f} to {bout['frame_range'][1] / fps:.2f} s
- Result: {result_text}
- Left Fencer:
  - Init Time: {bout['left_data']['first_step']['init_time']:.2f} s ({'Fast' if bout['left_data']['first_step']['is_fast'] else 'Slow'})
  - Init Velocity: {bout['left_data']['first_step']['velocity']:.2f}
  - Init Acceleration: {bout['left_data']['first_step']['acceleration']:.2f}
  - Advance Intervals: {bout['left_data']['advance_sec']}
  - Pause/Retreat Intervals: {bout['left_data']['pause_sec']}
  - Arm Extension Intervals: {bout['left_data']['arm_extensions_sec']}
  - Has Lunge: {bout['left_data']['has_launch']}
  - Lunge Frame: {bout['left_data']['launch_frame']} ({'N/A' if bout['left_data']['launch_frame'] is None or bout['left_data']['launch_frame'] == -1 else f"{bout['left_data']['launch_frame'] / fps:.2f} s"})
  - Velocity: {bout['left_data']['velocity']:.2f}
  - Acceleration: {bout['left_data']['acceleration']:.2f}
  - Latest Pause/Retreat End: {left_pause_end}
  - Advance Ratio: {bout['left_data']['advance_ratio']:.2f}
  - Pause Ratio: {bout['left_data']['pause_ratio']:.2f}
  - Arm Extension Freq: {bout['left_data']['arm_extension_freq']}
  - Avg Arm Extension Duration: {bout['left_data']['avg_arm_extension_duration']:.2f} s
  - Lunge Promptness: {left_launch_promptness} s
  - Footwork Data: {left_steps}
  {left_metrics}
- Right Fencer:
  - Init Time: {bout['right_data']['first_step']['init_time']:.2f} s ({'Fast' if bout['right_data']['first_step']['is_fast'] else 'Slow'})
  - Init Velocity: {bout['right_data']['first_step']['velocity']:.2f}
  - Init Acceleration: {bout['right_data']['first_step']['acceleration']:.2f}
  - Advance Intervals: {bout['right_data']['advance_sec']}
  - Pause/Retreat Intervals: {bout['right_data']['pause_sec']}
  - Arm Extension Intervals: {bout['right_data']['arm_extensions_sec']}
  - Has Lunge: {bout['right_data']['has_launch']}
  - Lunge Frame: {bout['right_data']['launch_frame']} ({'N/A' if bout['right_data']['launch_frame'] is None or bout['right_data']['launch_frame'] == -1 else f"{bout['right_data']['launch_frame'] / fps:.2f} s"})
  - Velocity: {bout['right_data']['velocity']:.2f}
  - Acceleration: {bout['right_data']['acceleration']:.2f}
  - Latest Pause/Retreat End: {right_pause_end}
  - Advance Ratio: {bout['right_data']['advance_ratio']:.2f}
  - Pause Ratio: {bout['right_data']['pause_ratio']:.2f}
  - Arm Extension Freq: {bout['right_data']['arm_extension_freq']}
  - Avg Arm Extension Duration: {bout['right_data']['avg_arm_extension_duration']:.2f} s
  - Lunge Promptness: {right_launch_promptness} s
  - Footwork Data: {right_steps}
  {right_metrics}
"""
        bout_sections.append(bout_section)

    fencer_sections = []
    for fencer_id, data in fencer_data.items():
        avg_launch_promptness = f"{data['avg_launch_promptness']:.2f}" if data['avg_launch_promptness'] != float('inf') else 'N/A'
        all_steps = []
        for bout in data['bouts']:
            side = 'left' if 'Fencer_Left' in fencer_id else 'right'
            steps = bout['metrics'].get('steps', [])
            all_steps.extend(steps)
        avg_step_duration = np.mean([s['Duration (s)'] for s in all_steps]) if all_steps else 0
        avg_step_speed = np.mean([s['Speed (units/s)'] for s in all_steps]) if all_steps else 0
        step_frequency = len(all_steps) / data['total_bouts'] if data['total_bouts'] > 0 else 0

        fencer_section = f"""
**Fencer {fencer_id}**:
- Total Bouts: {data['total_bouts']}
- Avg Init Time: {data['avg_first_step_init']:.2f} s
- Avg Init Velocity: {data['avg_first_step_velocity']:.2f}
- Avg Init Acceleration: {data['avg_first_step_acceleration']:.2f}
- Avg Velocity: {data['avg_velocity']:.2f} (Std Dev: {data['std_velocity']:.2f})
- Avg Acceleration: {data['avg_acceleration']:.2f} (Std Dev: {data['std_acceleration']:.2f})
- Avg Advance Ratio: {data['avg_advance_ratio']:.2f}
- Avg Pause Ratio: {data['avg_pause_ratio']:.2f}
- Total Arm Extensions: {data['total_arm_extensions']}
- Avg Arm Extension Duration: {data['avg_arm_extension_duration']:.2f} s
- Avg Lunge Promptness: {avg_launch_promptness} s
- Attack Frequency (>80 frames): {data['attacking_ratio']:.2f}
- Avg Footwork Duration: {avg_step_duration:.2f} s
- Avg Footwork Speed: {avg_step_speed:.2f} units/s
- Footwork Frequency: {step_frequency:.2f} steps/bout
"""
        fencer_sections.append(fencer_section)

    prompt = prompt.format(
        num_bouts=len(bout_data),
        fps=fps
    ) + "\n**Bout Details**:\n" + "\n".join(bout_sections) + "\n**Fencer Performance Summary**:\n" + "\n".join(fencer_sections)

    return prompt

def judge_bout_winner(bout, fps=30):
    """
    Judges the winner of a single bout based on the provided logic.
    Returns a dictionary with winner, confidence, and reasoning.
    """
    total_frames = bout['frame_range'][1] - bout['frame_range'][0] + 1
    left = bout['left_data']
    right = bout['right_data']
    video_angle = bout.get('video_angle')

    def get_winner(reasoning, winner, confidence):
        return {'winner': winner, 'confidence': confidence, 'reasoning': reasoning}

    if total_frames > 60:
        left_pause_end = left.get('latest_pause_retreat_end', -1)
        right_pause_end = right.get('latest_pause_retreat_end', -1)

        if left_pause_end != -1 and right_pause_end != -1:
            if left_pause_end < right_pause_end:
                return get_winner("Left fencer finished pause/retreat earlier and re-initiated the attack, thus gaining priority.", "left", 0.8)
            elif right_pause_end < left_pause_end:
                return get_winner("Right fencer finished pause/retreat earlier and re-initiated the attack, thus gaining priority.", "right", 0.8)
            else: # Equal pause end times, fall back to velocity/acceleration
                pass
        
        if left.get('velocity', 0) > right.get('velocity', 0):
            return get_winner("With no clear pause/retreat difference, the left fencer's higher overall velocity suggests a more decisive attack.", "left", 0.7)
        elif right.get('velocity', 0) > left.get('velocity', 0):
            return get_winner("With no clear pause/retreat difference, the right fencer's higher overall velocity suggests a more decisive attack.", "right", 0.7)
        else: # Equal velocity, check acceleration
            if left.get('acceleration', 0) > right.get('acceleration', 0):
                return get_winner("Velocities were comparable, but the left fencer's higher acceleration indicates a more explosive attack.", "left", 0.65)
            elif right.get('acceleration', 0) > left.get('acceleration', 0):
                return get_winner("Velocities were comparable, but the right fencer's higher acceleration indicates a more explosive attack.", "right", 0.65)

    else: # total_frames <= 60
        launch_relevant = video_angle not in ['left', 'right']
        left_launch = left.get('has_launch', False)
        right_launch = right.get('has_launch', False)

        if launch_relevant:
            if left.get('velocity', 0) >= 2 * right.get('velocity', 0) and right.get('velocity', 0) > 0:
                 return get_winner("Left fencer's velocity was significantly dominant, likely overwhelming the opponent.", "left", 0.7)
            if right.get('velocity', 0) >= 2 * left.get('velocity', 0) and left.get('velocity', 0) > 0:
                 return get_winner("Right fencer's velocity was significantly dominant, likely overwhelming the opponent.", "right", 0.7)

            if left_launch and not right_launch:
                return get_winner("Left fencer lunged in the final phase while the opponent did not, indicating a completed attack.", "left", 0.65)
            if right_launch and not left_launch:
                return get_winner("Right fencer lunged in the final phase while the opponent did not, indicating a completed attack.", "right", 0.65)
            if left_launch and right_launch:
                if left.get('launch_frame', -1) < right.get('launch_frame', -1):
                    return get_winner("Both fencers lunged, but the left fencer initiated earlier, seizing the opportunity.", "left", 0.6)
                if right.get('launch_frame', -1) < left.get('launch_frame', -1):
                    return get_winner("Both fencers lunged, but the right fencer initiated earlier, seizing the opportunity.", "right", 0.6)
        
        # Fallback if launches are irrelevant or indecisive
        has_pauses = bool(left.get('pause_sec')) or bool(right.get('pause_sec'))
        if has_pauses:
            left_v = left.get('post_pause_velocity') if left.get('post_pause_velocity') != 0 else left.get('velocity')
            right_v = right.get('post_pause_velocity') if right.get('post_pause_velocity') != 0 else right.get('velocity')
            left_a = left.get('post_pause_acceleration') if left.get('post_pause_acceleration') != 0 else left.get('acceleration')
            right_a = right.get('post_pause_acceleration') if right.get('post_pause_acceleration') != 0 else right.get('acceleration')
            # Safeguard against missing acceleration values to prevent TypeError during comparison
            left_a = 0 if left_a is None else left_a
            right_a = 0 if right_a is None else right_a
            
            left_ext = max([e[1] for e in left['arm_extensions_sec']]) if left['arm_extensions_sec'] else -1
            right_ext = max([e[1] for e in right['arm_extensions_sec']]) if right['arm_extensions_sec'] else -1

            if left_a > right_a:
                return get_winner("After a pause, the left fencer's higher acceleration suggests a stronger offensive intent.", "left", 0.6)
            elif right_a > left_a:
                return get_winner("After a pause, the right fencer's higher acceleration suggests a stronger offensive intent.", "right", 0.6)
            else: # Equal acceleration
                if left_ext < right_ext and left_ext != -1:
                     return get_winner("Accelerations were similar, but the left fencer's earlier arm extension may have secured priority.", "left", 0.55)
                if right_ext < left_ext and right_ext != -1:
                     return get_winner("Accelerations were similar, but the right fencer's earlier arm extension may have secured priority.", "right", 0.55)
        else: # No pauses
            left_v = left.get('velocity', 0)
            right_v = right.get('velocity', 0)
            left_ext = max([e[1] for e in left['arm_extensions_sec']]) if left['arm_extensions_sec'] else -1
            right_ext = max([e[1] for e in right['arm_extensions_sec']]) if right['arm_extensions_sec'] else -1

            if left_v > right_v:
                return get_winner("Without a pause, the left fencer had a higher overall velocity.", "left", 0.55)
            elif right_v > left_v:
                return get_winner("Without a pause, the right fencer had a higher overall velocity.", "right", 0.55)
            else: # Equal velocity
                 if left_ext < right_ext and left_ext != -1:
                     return get_winner("Velocities were comparable, but the left fencer's earlier arm extension might have gained priority.", "left", 0.5)
                 if right_ext < left_ext and right_ext != -1:
                     return get_winner("Velocities were comparable, but the right fencer's earlier arm extension might have gained priority.", "right", 0.5)

    return get_winner("Metrics for both fencers are too close to make a determination.", "undetermined", 0.5)


def generate_individual_bout_analysis(bout, judgement, fps=30):
    """
    Generate a detailed analysis for a single bout using GPT.
    """
    bout_result = bout['bout_result']
    if bout_result and bout_result not in ['skip', 'undetermined']:
        winner_text = f"The designated winner of this bout is the **{'Left' if bout_result == 'left' else 'Right'} Fencer**."
    else:
        winner_text = f"Based on data analysis, we infer the **{'Left' if judgement['winner'] == 'left' else 'Right'} Fencer** won with {judgement['confidence']:.0%} confidence. Reason: {judgement['reasoning']}"

    # Sanitize bout data before dumping to JSON
    # sanitized_bout = convert_numpy_types(bout)

    # Extract tactical analysis data
    left_attacks = []
    right_attacks = []
    left_defenses = []
    right_defenses = []
    
    for side in ['left', 'right']:
        side_data = bout.get(f'{side}_data', {})
        interval_analysis = side_data.get('interval_analysis', {})
        
        # Extract attack analyses
        advance_analyses = interval_analysis.get('advance_analyses', [])
        for analysis in advance_analyses:
            attack_info = {
                'type': analysis.get('attack_classification', {}).get('attack_type', 'Unknown'),
                'tempo_quality': analysis.get('tempo_analysis', {}).get('tempo_quality', 'Unknown'),
                'tempo_variation': analysis.get('tempo_analysis', {}).get('tempo_variation', 0),
                'effectiveness': analysis.get('attack_classification', {}).get('effectiveness', 'Unknown'),
                'interval': analysis.get('interval', [0, 0])
            }
            if side == 'left':
                left_attacks.append(attack_info)
            else:
                right_attacks.append(attack_info)
        
        # Extract defense analyses
        retreat_analyses = interval_analysis.get('retreat_analyses', [])
        for analysis in retreat_analyses:
            defense_info = {
                'reaction_type': analysis.get('defensive_classification', {}).get('reaction_type', 'Unknown'),
                'composure': analysis.get('defensive_classification', {}).get('composure', 'Unknown'),
                'distance_management': analysis.get('distance_analysis', {}).get('distance_management_quality', 'Unknown'),
                'interval': analysis.get('interval', [0, 0])
            }
            if side == 'left':
                left_defenses.append(defense_info)
            else:
                right_defenses.append(defense_info)

    prompt = f"""
    You are an AI saber fencing analyst with access to advanced tactical classification systems, analyzing an individual bout.
    **Background**: Both fencers score (no single-light), parry-ripostes cannot be detected. Your analysis leverages detailed attack/defense classification and the actual bout outcome.

    **Enhanced Bout Data**:
    {json.dumps(create_bout_data_for_gpt(bout), indent=2, ensure_ascii=False)}
    
    **Tactical Analysis Data**:
    **Left Fencer Attacks**: {json.dumps(left_attacks, indent=2)}
    **Right Fencer Attacks**: {json.dumps(right_attacks, indent=2)}
    **Left Fencer Defenses**: {json.dumps(left_defenses, indent=2)}
    **Right Fencer Defenses**: {json.dumps(right_defenses, indent=2)}

    **Bout Outcome**: {winner_text}

    **Enhanced Analysis Task**:
    使用战术分类数据和回合结果，提供中文的综合分析：

    1.  **战术叙述**：使用分类的攻击和防守类型将回合描述为战略交锋。
        -   引用具体攻击类型（直接、复合、基于佯攻、准备性）及其时机
        -   描述防守反应（控制性后退、反击等）及其有效性
        -   解释节奏质量和变化如何影响交锋
        -   将战术选择与回合结果联系起来

    2.  **战略有效性分析**：评估哪些战术有效及其原因。
        -   分析攻击类型成功：每位击剑手的哪些攻击有效？
        -   评估防守表现：每位击剑手应对压力的表现如何？
        -   评估节奏控制：谁更好地管理了节奏？
        -   将战术模式与最终结果联系

    3.  **生物力学和技术评估**：使用冲刺和伸展数据评估执行质量。
        -   评论冲刺力学（足部速度、臀部参与、时机）
        -   评估手臂伸展模式及其战术含义
        -   评估防守阶段的距离管理

    4.  **表现洞察和建议**：基于结果关联提供具体反馈。
        -   **获胜者优势**：哪些战术要素促成了胜利？
        -   **改进领域**：什么战术调整可能改变结果？
        -   **战略建议**：对类似未来情况的具体战术建议

    专注于将详细的战术数据与回合结果连接，以提供关于实际有效策略的可操作洞察。
    """
    try:
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "您是一位专业的佩剑击剑分析师，提供单回合的详细中文战术分析。"},
                {"role": "user", "content": prompt}
            ],
            temperature=0.7,
            max_tokens=1500
        )
        return response.choices[0].message.content
    except Exception as e:
        logging.error(f"Error generating individual bout analysis for match {bout.get('match_idx')}: {e}")
        return "Individual bout analysis is unavailable due to an API error."

def generate_general_analysis_prompt(bout_data, fencer_data, left_data, right_data, fps=30):
    """Generate prompt for the general analysis report, including the 8 specific graph analyses."""
    
    # Build comprehensive statistics from the extracted data
    analysis_summary = {}
    
    # Process left fencer data
    analysis_summary['left_fencer'] = {
        'attack_types': len(left_data['attack_types']),
        'tempo_types': len(left_data['tempo_types']),
        'attack_distances': len(left_data['attack_distances']),

        'counter_opportunities': len(left_data['counter_opportunities']),
        'retreat_quality': len(left_data['retreat_quality']),
        'defensive_quality': len(left_data['defensive_quality']),
        'bout_outcomes': len(left_data['bout_outcomes'])
    }
    
    # Process right fencer data
    analysis_summary['right_fencer'] = {
        'attack_types': len(right_data['attack_types']),
        'tempo_types': len(right_data['tempo_types']),
        'attack_distances': len(right_data['attack_distances']),

        'counter_opportunities': len(right_data['counter_opportunities']),
        'retreat_quality': len(right_data['retreat_quality']),
        'defensive_quality': len(right_data['defensive_quality']),
        'bout_outcomes': len(right_data['bout_outcomes'])
    }
    
    # Count bout results by winner_side
    bout_results_summary = {'left': 0, 'right': 0, 'undetermined': 0}
    for bout in bout_data:
        winner = bout.get('winner_side', 'undetermined')
        if winner in bout_results_summary:
            bout_results_summary[winner] += 1
        else:
            bout_results_summary['undetermined'] += 1
    
    prompt = """
    You are an AI fencing analyst specializing in saber, tasked with evaluating multiple bouts to provide a comprehensive, data-driven analysis of fencer performance focused on ATTACK and DEFENSE EFFECTIVENESS. You have access to detailed tactical analysis that correlates specific actions with bout outcomes, allowing you to identify what strategies actually work for each fencer. Your analysis should emphasize tactical effectiveness, strategic patterns, and evidence-based recommendations. All output should be in English.

    **Analysis Data Summary**:
    - **Bouts Analyzed**: {num_bouts} bouts with winner determination
    - **Bout Winners**: {bout_results}
    - **Analysis Coverage**: {analysis_summary}
    
    **8 Key Analysis Areas**:
    1. **Attack Type Victory Analysis**: Count and success rate of different attack types
    2. **Tempo Type Victory Analysis**: Count and success rate of different tempo patterns
    3. **Good Attack Distance Analysis**: Percentage of attacks launched at optimal distance
    4. **Dangerous Close Frame Analysis**: Percentage of advance intervals with dangerous proximity
    5. **Counter Opportunity Analysis**: Counter opportunities identified, used, and missed
    6. **Retreat Quality Analysis**: Safe distance maintenance and spacing consistency
    7. **Defensive Quality Analysis**: Good vs poor defensive actions
    8. **Bout Outcome Analysis**: Wins through attack style vs retreat style

    Please use the following terms throughout your response to refer to different actions and intentions.
    - **Compound Attack**: An attack with multiple feints or blade actions (e.g., feint-cut, feint-disengage) to deceive the opponent.
    - **Simultaneous Attack**: Both fencers attack at the same time, with priority determining the score.
    - **Priority**: A rule that gives the right-of-way to the fencer who correctly executes an attack, inferred from arm extension timing and attack continuity.
    - **Distance Management**: Controlling the lunge, retreat, preparation, or breaking distance to induce an opponent or close distance unexpectedly.
    - **Footwork**: Includes advance-lunge (a quick step and lunge) and flèche (a running attack, restricted in modern rules).
    - **Breaking Distance**: Maintaining a distance just outside the opponent's reach to provoke a premature attack, then countering.
    - **Feint**: A deceptive action or blade movement, such as a disengage (slipping under the opponent's parry) or a cut-over (whipping the blade over the opponent's), to bypass their defense.
    - **First and Second Intention**: First intention is a direct attack to score; second intention is to induce a reaction (e.g., a parry, a counter-attack) to set up a subsequent action.
    - **Tempo Control**: Varying the speed of an attack or footwork to disrupt the opponent's rhythm, e.g., slowing the preparation then accelerating into a lunge.

    **输出要求（中文）**：
    
    ## 攻击有效性报告
    
    ### 左侧击剑手攻击分析：
    - **攻击类型有效性**：分析使用的每种攻击类型的成功率
    - **获胜攻击档案**：什么攻击策略导致胜利？
    - **攻击效率**：成功攻击的速度、时机和执行质量
    - **建议**：基于有效性，应强调/避免哪些攻击类型
    
    ### 右侧击剑手攻击分析：
    - **攻击类型有效性**：分析使用的每种攻击类型的成功率
    - **获胜攻击档案**：什么攻击策略导致胜利？
    - **攻击效率**：成功攻击的速度、时机和执行质量
    - **建议**：基于有效性，应强调/避免哪些攻击类型
    
    ## 防守有效性报告
    
    ### 左侧击剑手防守分析：
    - **防守反应有效性**：不同防守反应的成功率
    - **反击机会**：他们在防守情况下的利用能力如何
    - **距离管理**：后退和位置决策的质量
    - **建议**：如何提高防守有效性和反击能力
    
    ### 右侧击剑手防守分析：
    - **防守反应有效性**：不同防守反应的成功率
    - **反击机会**：他们在防守情况下的利用能力如何
    - **距离管理**：后退和位置决策的质量
    - **建议**：如何提高防守有效性和反击能力
    
    ## 战略对比与洞察
    - **获胜模式对比**：每位击剑手通常如何获胜（攻击vs防守）
    - **战术对决**：相互对阵时的优势/劣势
    - **关键成功因素**：什么区分了获胜表现和失败
    - **训练优先级**：每位击剑手基于证据的重点领域
    """
    fencer_sections = []
    for fencer_id, data in fencer_data.items():
        avg_launch_promptness = f"{data['avg_launch_promptness']:.2f}" if data['avg_launch_promptness'] != float('inf') else 'N/A'
        all_steps = []
        for bout in data['bouts']:
            side = 'left' if 'Fencer_Left' in fencer_id else 'right'
            steps = bout['metrics'].get('steps', [])
            all_steps.extend(steps)
        avg_step_duration = np.mean([s['Duration (s)'] for s in all_steps]) if all_steps else 0
        avg_step_speed = np.mean([s['Speed (units/s)'] for s in all_steps]) if all_steps else 0
        step_frequency = len(all_steps) / data['total_bouts'] if data['total_bouts'] > 0 else 0

        fencer_section = f"""
**Fencer {fencer_id}**:
- Total Bouts: {data['total_bouts']}
- Average Start Time: {data['avg_first_step_init']:.2f} sec
- Average Initial Velocity: {data['avg_first_step_velocity']:.2f}
- Average Initial Acceleration: {data['avg_first_step_acceleration']:.2f}
- Average Velocity: {data['avg_velocity']:.2f} (Std Dev: {data['std_velocity']:.2f})
- Average Acceleration: {data['avg_acceleration']:.2f} (Std Dev: {data['std_acceleration']:.2f})
- Average Advance Ratio: {data['avg_advance_ratio']:.2f}
- Average Pause Ratio: {data['avg_pause_ratio']:.2f}
- Total Arm Extensions: {data['total_arm_extensions']}
- Average Arm Extension Duration: {data['avg_arm_extension_duration']:.2f} sec
- Average Lunge Promptness: {avg_launch_promptness} sec
- Attack Frequency (>80 frames): {data['attacking_ratio']:.2f}
- Average Footwork Duration: {avg_step_duration:.2f} sec
- Average Footwork Speed: {avg_step_speed:.2f} units/sec
- Footwork Frequency: {step_frequency:.2f} steps/bout
"""
        fencer_sections.append(fencer_section)

    prompt = prompt.format(
        num_bouts=len(bout_data),
        fps=fps,
        bout_results=bout_results_summary,
        analysis_summary=analysis_summary
    ) + "".join(fencer_sections)

    return prompt



def main(analysis_dir, match_data_dir, output_dir="./result/fencer_analysis", fps=30, show_dfs=False):
    logging.info(f"Starting fencer analysis with analysis_dir: {analysis_dir}, match_data_dir: {match_data_dir}, output_dir: {output_dir}")
    
    bout_data = load_bout_data(analysis_dir, match_data_dir)
    if not bout_data:
        logging.error("No bout data loaded, exiting")
        return
    
    if show_dfs:
        logging.info("Showing head of left_x_df for the first bout as requested:")
        if bout_data:
            print(bout_data[0]['left_x_df'].head())
        else:
            logging.info("No bout data to show.")
    
    bout_data = compute_derived_metrics(bout_data, fps)
    
    fencer_data = aggregate_fencer_data(bout_data)
    
    # Add winner_side tagging BEFORE generating plots
    for bout in bout_data:
        bout_result = bout.get('bout_result')
        if not bout_result or bout_result in ['skip', 'undetermined']:
            judgement = judge_bout_winner(bout, fps)
            bout['judgement'] = judgement
            bout['winner_side'] = judgement['winner']
        else:
            bout['winner_side'] = bout_result
    
    # Classify touch categories for all bouts
    bout_classifications, touch_stats = classify_all_bouts(bout_data)
    logging.info(f"Generated touch category classifications and statistics")
    
    # Process In-Box analysis from existing match analysis files with winner information
    # match_analysis is a sibling of the fencer_analysis output directory
    match_analysis_dir = os.path.join(os.path.dirname(output_dir), 'match_analysis')
    
    # Use comprehensive In-Box analysis with winner correlations
    from your_scripts.inbox_comprehensive_analysis import process_inbox_bouts_with_winners
    inbox_data = process_inbox_bouts_with_winners(match_analysis_dir, bout_classifications)
    
    # Integrate In-Box details with touch classifications
    enhanced_classifications = integrate_inbox_with_touch_classification(bout_classifications, bout_data, fps)
    
    # Update touch stats with comprehensive In-Box details
    touch_stats['inbox_analysis'] = inbox_data
    logging.info(f"Processed {inbox_data.get('total_inbox_bouts', 0)} In-Box bouts with win/loss analysis")
    
    # Process attack bouts with comprehensive analysis
    attack_data = process_attack_bouts_with_winners(match_analysis_dir, bout_classifications)
    touch_stats['attack_analysis'] = attack_data
    logging.info(f"Processed {attack_data.get('total_attack_bouts', 0)} attack bouts with win/loss analysis")
    
    # Process defense bouts with comprehensive analysis  
    defense_data = process_defense_bouts_with_winners(match_analysis_dir, bout_classifications)
    touch_stats['defense_analysis'] = defense_data
    logging.info(f"Processed {defense_data.get('total_defense_bouts', 0)} defense bouts with win/loss analysis")
    
    # Graph generation disabled - no longer needed for main view
    # chart_analysis = generate_plots(fencer_data, os.path.join(output_dir, 'plots'))
    chart_analysis = {}
    
    # Touch category visualizations disabled - no longer needed
    # try:
    #     touch_chart_paths = create_touch_category_charts(touch_stats, os.path.join(output_dir, 'touch_category_charts'))
    #     touch_summary = generate_touch_category_summary(touch_stats)
    #
    #     # In-Box charts are now generated within the comprehensive analysis
    #     # Get chart paths from inbox_data
    #     inbox_chart_paths = inbox_data.get('chart_paths', {})
    #     touch_chart_paths.update(inbox_chart_paths)
    #
    #     # Generate attack charts
    #     attack_chart_paths = create_comprehensive_attack_charts(
    #         attack_data.get('bouts', []),
    #         os.path.join(output_dir, 'touch_category_charts', 'attack')
    #     )
    #     touch_chart_paths.update({f'attack_{k}': v for k, v in attack_chart_paths.items()})
    #
    #     # Generate defense charts
    #     defense_chart_paths = create_comprehensive_defense_charts(
    #         defense_data.get('bouts', []),
    #         os.path.join(output_dir, 'touch_category_charts', 'defense')
    #     )
    #     touch_chart_paths.update({f'defense_{k}': v for k, v in defense_chart_paths.items()})
    #
    #     logging.info(f"Generated touch category charts including attack/defense analysis")
    # except Exception as e:
    #     logging.error(f"Error generating touch category visualizations: {e}")
    touch_chart_paths = {}
    touch_summary = ""
    
    # Advanced plots generation disabled - no longer needed for main view
    # try:
    #     left_data, right_data = extract_fencer_analysis_data(bout_data)
    #     logging.info(f"Extracted analysis data for both fencers")
    #
    #     plot_results = save_all_plots(left_data, right_data,
    #                                   os.path.join(output_dir, 'advanced_plots'))
    #     new_plot_files = plot_results.get('plot_files', {})
    #     graph_analysis = plot_results.get('analysis', {}).get('graph_analysis', {})
    #     logging.info(f"Generated all 8 analysis plots and graph analysis for both fencers")
    # except Exception as e:
    #     logging.error(f"Error generating analysis plots: {str(e)}\n{traceback.format_exc()}")
    new_plot_files = {}
    graph_analysis = {}
    
    # Build a concise cross-bout Chinese summary so 'analysis' is never None
    def build_cross_bout_summary(fd: dict, bouts: list) -> str:
        try:
            total_bouts = len(bouts)
            # Determine side keys if available
            left_key = next((k for k in fd.keys() if str(k).lower().endswith('left')), None)
            right_key = next((k for k in fd.keys() if str(k).lower().endswith('right')), None)
            # Fallback: pick any two keys deterministically
            keys = list(fd.keys())
            if not left_key and keys:
                left_key = keys[0]
            if not right_key and len(keys) > 1:
                right_key = keys[1]

            def mget(obj: dict, key: str, default: float = 0.0) -> float:
                try:
                    return float(obj.get(key, default) or default)
                except Exception:
                    return default

            left = fd.get(left_key, {}) if left_key in fd else {}
            right = fd.get(right_key, {}) if right_key in fd else {}

            # Key metrics
            l_first = mget(left, 'avg_first_step_init')
            r_first = mget(right, 'avg_first_step_init')
            l_vel = mget(left, 'avg_velocity')
            r_vel = mget(right, 'avg_velocity')
            l_acc = mget(left, 'avg_acceleration')
            r_acc = mget(right, 'avg_acceleration')
            l_adv = mget(left, 'avg_advance_ratio')
            r_adv = mget(right, 'avg_advance_ratio')
            l_pause = mget(left, 'avg_pause_ratio')
            r_pause = mget(right, 'avg_pause_ratio')

            # Winner distribution if available
            left_wins = 0
            right_wins = 0
            for b in bouts:
                winner = b.get('winner_side') or (b.get('judgement') or {}).get('winner')
                if not winner:
                    continue
                wl = str(winner).lower()
                if 'left' in wl:
                    left_wins += 1
                elif 'right' in wl:
                    right_wins += 1

            lines = []
            lines.append(f"本次视频共包含{total_bouts}个回合；左侧击剑手胜{left_wins}回合，右侧击剑手胜{right_wins}回合。")
            if l_first or r_first:
                faster = '左侧' if (l_first and r_first and l_first < r_first) else ('右侧' if (l_first and r_first and r_first < l_first) else '两侧')
                lines.append(f"首步时机对比：{faster}更快（左{l_first:.2f}s vs 右{r_first:.2f}s）。")
            if l_vel or r_vel:
                better = '左侧' if l_vel > r_vel else ('右侧' if r_vel > l_vel else '两侧')
                lines.append(f"平均速度：{better}更高（左{l_vel:.2f} m/s vs 右{r_vel:.2f} m/s）。")
            if l_acc or r_acc:
                better = '左侧' if l_acc > r_acc else ('右侧' if r_acc > l_acc else '两侧')
                lines.append(f"平均加速度：{better}更强（左{l_acc:.2f} m/s² vs 右{r_acc:.2f} m/s²）。")
            if l_adv or r_adv:
                better = '左侧' if l_adv > r_adv else ('右侧' if r_adv > l_adv else '两侧')
                lines.append(f"前进压力：{better}更主动（左{l_adv:.2f} vs 右{r_adv:.2f}）。")
            if l_pause or r_pause:
                better = '左侧' if l_pause < r_pause else ('右侧' if r_pause < l_pause else '两侧')
                lines.append(f"节奏停顿控制：{better}更稳定（左{l_pause:.2f} vs 右{r_pause:.2f}）。")

            lines.append("训练建议：保持强项，针对弱项进行专项训练（首步爆发、距离控制与节奏变化）。")
            return "\n".join(lines)
        except Exception:
            return "本次视频已生成分析图表与指标摘要。"

        
    cross_bout_text = build_cross_bout_summary(fencer_data, bout_data)

    save_analysis(fencer_data, cross_bout_text, bout_data, output_dir, fps, chart_analysis, new_plot_files, graph_analysis, touch_stats, touch_summary, touch_chart_paths, enhanced_classifications['classifications'])

    
def save_analysis(fencer_data, cross_bout_analysis, bout_data, output_dir, fps, chart_analysis=None, advanced_plot_files=None, graph_analysis=None, touch_stats=None, touch_summary=None, touch_chart_paths=None, bout_classifications=None):
    # The convert_numpy_types function is now at the module level
    logging.info(f"Saving analysis to {output_dir}")
    os.makedirs(output_dir, exist_ok=True)
    
    for fencer_id, data in fencer_data.items():
        fencer_output = {
            'fencer_id': fencer_id,
            'metrics': {
                'avg_first_step_init': data['avg_first_step_init'],
                'avg_first_step_velocity': data['avg_first_step_velocity'],
                'avg_first_step_acceleration': data['avg_first_step_acceleration'],
                'avg_velocity': data['avg_velocity'],
                'std_velocity': data['std_velocity'],
                'avg_acceleration': data['avg_acceleration'],
                'std_acceleration': data['std_acceleration'],
                'avg_advance_ratio': data['avg_advance_ratio'],
                'avg_pause_ratio': data['avg_pause_ratio'],
                'total_arm_extensions': data['total_arm_extensions'],
                'avg_arm_extension_duration': data['avg_arm_extension_duration'],
                'avg_launch_promptness': data['avg_launch_promptness'],
                'attacking_ratio': data['attacking_ratio'],
                'avg_right_of_way_score': data['avg_right_of_way_score']
            },
            'bouts': data['bouts']
        }
        fencer_output = convert_numpy_types(fencer_output)
        output_path = os.path.join(output_dir, f"fencer_{fencer_id}_analysis.json")
        try:
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(fencer_output, f, ensure_ascii=False, indent=4)
            logging.info(f"Saved fencer {fencer_id} analysis to {output_path}")
        except Exception as e:
            logging.error(f"Error saving fencer {fencer_id} JSON: {str(e)}\n{traceback.format_exc()}")
    
    cross_bout_output = {
        'fencers': list(fencer_data.keys()),
        'analysis': cross_bout_analysis,  # This will be None now
        'chart_analysis': chart_analysis or {},  # New chart analysis
        'graph_analysis': graph_analysis or {},  # New graph analysis  
        'advanced_plot_files': advanced_plot_files or {},
        'touch_category_analysis': {
            'statistics': touch_stats or {},
            'summary': touch_summary or '',
            'chart_paths': touch_chart_paths or {}
        }
    }
    output_path = os.path.join(output_dir, 'cross_bout_analysis.json')
    try:
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(convert_numpy_types(cross_bout_output), f, ensure_ascii=False, indent=4)
        logging.info(f"Saved cross-bout analysis to {output_path}")
    except Exception as e:
        logging.error(f"Error saving cross-bout JSON: {str(e)}\n{traceback.format_exc()}")
    
    bout_summaries = []
    
    # Create a lookup for bout classifications
    bout_class_lookup = {bc['match_idx']: bc for bc in bout_classifications} if bout_classifications else {}
    
    for bout in bout_data:
        match_idx = bout['match_idx']
        total_frames = bout['frame_range'][1] - bout['frame_range'][0] + 1
        is_long_bout = total_frames > 80
        upload_id = bout['upload_id']
        
        bout_result = Bout.query.filter_by(upload_id=upload_id, match_idx=match_idx).first()
        result_text = bout_result.result if bout_result and bout_result.result else None

        left_pause_end = bout['left_data']['latest_pause_retreat_end'] / fps if bout['left_data']['latest_pause_retreat_end'] != -1 else None
        right_pause_end = bout['right_data']['latest_pause_retreat_end'] / fps if bout['right_data']['latest_pause_retreat_end'] != -1 else None
        left_launch_promptness = bout['left_data']['launch_promptness'] if bout['left_data']['launch_promptness'] != float('inf') else None
        right_launch_promptness = bout['right_data']['launch_promptness'] if bout['right_data']['launch_promptness'] != float('inf') else None

        # Get touch classification for this bout
        bout_classification = bout_class_lookup.get(match_idx, {})
        
        summary = {
            'match_idx': match_idx,
            'frame_range': [bout['frame_range'][0] / fps, bout['frame_range'][1] / fps],
            'type': 'Attack-Defense' if is_long_bout else 'In-Box',
            'total_frames': total_frames,
            'result': result_text,
            'judgement': bout.get('judgement'),
            'individual_analysis': bout.get('individual_analysis'),
            'touch_classification': {
                'left_category': bout_classification.get('left_category', 'unknown'),
                'right_category': bout_classification.get('right_category', 'unknown'),
                'winner': bout_classification.get('winner', 'undetermined')
            },
            'left_data': {  # Changed from 'left_fencer'
                'first_step': {
                    'init_time': bout['left_data']['first_step']['init_time'],
                    'is_fast': bout['left_data']['first_step']['is_fast'],
                    'velocity': bout['left_data']['first_step']['velocity'],
                    'acceleration': bout['left_data']['first_step']['acceleration']
                },
                'advance_intervals': bout['left_data']['advance_sec'],
                'pause_intervals': bout['left_data']['pause_sec'],
                'arm_extensions': bout['left_data']['arm_extensions_sec'],
                'arm_extension_freq': bout['left_data']['arm_extension_freq'],
                'has_launch': bout['left_data']['has_launch'],
                'launch_frame': bout['left_data']['launch_frame'] / fps if bout['left_data']['launch_frame'] != -1 else None,
                'velocity': bout['left_data']['velocity'],
                'acceleration': bout['left_data']['acceleration'],
                'latest_pause_end': left_pause_end,
                'advance_ratio': bout['left_data']['advance_ratio'],
                'pause_ratio': bout['left_data']['pause_ratio'],
                'avg_arm_extension_duration': bout['left_data']['avg_arm_extension_duration'],
                'launch_promptness': left_launch_promptness,
                'is_attacking': bout['left_data'].get('is_attacking', False) if is_long_bout else None,
                'first_pause_time': bout['left_data']['first_pause_time'] if not is_long_bout else None,
                'first_restart_time': bout['left_data']['first_restart_time'] if not is_long_bout else None,
                'post_pause_velocity': bout['left_data']['post_pause_velocity'] if not is_long_bout else None,
                'post_pause_acceleration': bout['left_data'].get('post_pause_acceleration') if not is_long_bout else None,
                'steps': bout['left_data'].get('steps', [])
            },
            'right_data': {  # Changed from 'right_fencer'
                'first_step': {
                    'init_time': bout['right_data']['first_step']['init_time'],
                    'is_fast': bout['right_data']['first_step']['is_fast'],
                    'velocity': bout['right_data']['first_step']['velocity'],
                    'acceleration': bout['right_data']['first_step']['acceleration']
                },
                'advance_intervals': bout['right_data']['advance_sec'],
                'pause_intervals': bout['right_data']['pause_sec'],
                'arm_extensions': bout['right_data']['arm_extensions_sec'],
                'arm_extension_freq': bout['right_data']['arm_extension_freq'],
                'has_launch': bout['right_data']['has_launch'],
                'launch_frame': bout['right_data']['launch_frame'] / fps if bout['right_data']['launch_frame'] != -1 else None,
                'velocity': bout['right_data']['velocity'],
                'acceleration': bout['right_data']['acceleration'],
                'latest_pause_end': right_pause_end,
                'advance_ratio': bout['right_data']['advance_ratio'],
                'pause_ratio': bout['right_data']['pause_ratio'],
                'avg_arm_extension_duration': bout['right_data']['avg_arm_extension_duration'],
                'launch_promptness': right_launch_promptness,
                'is_attacking': bout['right_data'].get('is_attacking', False) if is_long_bout else None,
                'first_pause_time': bout['right_data']['first_pause_time'] if not is_long_bout else None,
                'first_restart_time': bout['right_data']['first_restart_time'] if not is_long_bout else None,
                'post_pause_velocity': bout['right_data']['post_pause_velocity'] if not is_long_bout else None,
                'post_pause_acceleration': bout['right_data'].get('post_pause_acceleration') if not is_long_bout else None,
                'steps': bout['right_data'].get('steps', [])
            }
        }
        bout_summaries.append(summary)
        
    bout_summaries_output = {
        'fencers': list(fencer_data.keys()),
        'bouts': bout_summaries,
        'fps': fps,
        'touch_statistics': touch_stats or {}
    }
    bout_summaries_output = convert_numpy_types(bout_summaries_output)
    output_path = os.path.join(output_dir, 'bout_summaries.json')
    try:
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(bout_summaries_output, f, ensure_ascii=False, indent=4)
        logging.info(f"Saved bout summaries to {output_path}")
    except Exception as e:
        logging.error(f"Error saving bout summaries JSON: {str(e)}\n{traceback.format_exc()}")

    # Also write a minimal report.json so the report view can render
    try:
        # Build a report structure compatible with templates/report.html
        # Meta information (best effort)
        first_upload_id = None
        if isinstance(bout_data, list) and len(bout_data) > 0:
            first_upload_id = bout_data[0].get('upload_id')
        report_output = {
            'title': '分析报告',
            'meta': {
                'video_name': f"上传 {first_upload_id}" if first_upload_id is not None else '上传',
                'weapon': 'sabre'
            },
            'summary_text': cross_bout_analysis if isinstance(cross_bout_analysis, str) else '',
            'scores': {
                'left': {'overall': 0, 'offense': 0, 'defense': 0, 'tempo': 0, 'distance': 0},
                'right': {'overall': 0, 'offense': 0, 'defense': 0, 'tempo': 0, 'distance': 0}
            },
            'highlights': [],
            'outcomes': {
                'left_wins': 0,
                'right_wins': 0,
                'skips': 0
            },
            'kpis': {
                'left': {
                    'avg_velocity': fencer_data[list(fencer_data.keys())[0]].get('avg_velocity', 0) if fencer_data else 0,
                    'avg_acceleration': fencer_data[list(fencer_data.keys())[0]].get('avg_acceleration', 0) if fencer_data else 0,
                    'advance_ratio': fencer_data[list(fencer_data.keys())[0]].get('avg_advance_ratio', 0) if fencer_data else 0,
                    'pause_ratio': fencer_data[list(fencer_data.keys())[0]].get('avg_pause_ratio', 0) if fencer_data else 0,
                    'first_step_init': fencer_data[list(fencer_data.keys())[0]].get('avg_first_step_init', 0) if fencer_data else 0,
                    'launch_success_rate': 0
                },
                'right': {
                    'avg_velocity': fencer_data[list(fencer_data.keys())[-1]].get('avg_velocity', 0) if fencer_data else 0,
                    'avg_acceleration': fencer_data[list(fencer_data.keys())[-1]].get('avg_acceleration', 0) if fencer_data else 0,
                    'advance_ratio': fencer_data[list(fencer_data.keys())[-1]].get('avg_advance_ratio', 0) if fencer_data else 0,
                    'pause_ratio': fencer_data[list(fencer_data.keys())[-1]].get('avg_pause_ratio', 0) if fencer_data else 0,
                    'first_step_init': fencer_data[list(fencer_data.keys())[-1]].get('avg_first_step_init', 0) if fencer_data else 0,
                    'launch_success_rate': 0
                }
            },
            'graphs': {
                'bar_comparison': {'image': 'fencer_analysis/plots/bar_comparison.png'},
                'radar': {'image': 'fencer_analysis/plots/radar.png'}
            },
            'sections': {
                'overall': {
                    'bar_comparison': {
                        'left_bullets': [],
                        'right_bullets': []
                    },
                    'radar': {
                        'left_bullets': [],
                        'right_bullets': []
                    }
                },
                'attack': [],
                'defense': [],
                'attack_evaluation': {'left': [], 'right': []},
                'defense_evaluation': {'left': [], 'right': []}
            },
            'per_bout': [],
            'tags_summary': {
                'left_top': [],
                'right_top': []
            },
            'recommendations_left': [],
            'recommendations_right': [],
            'recommendations': {'offense': [], 'defense': []},
            'downloads': {'csv': []},
            'charts': {
                'plots': chart_analysis or {},
                'advanced_plots': advanced_plot_files or {},
                'touch_category': touch_chart_paths or {}
            },
            'touch_category_analysis': {
                'statistics': touch_stats or {},
                'summary': touch_summary or ''
            },
            'bouts': bout_summaries
        }
        # Compute outcomes based on bout_summaries if available
        try:
            left_wins = 0
            right_wins = 0
            skips = 0
            for b in bout_summaries:
                winner = (b.get('touch_classification') or {}).get('winner')
                if winner == 'left':
                    left_wins += 1
                elif winner == 'right':
                    right_wins += 1
                else:
                    skips += 1
            report_output['outcomes'] = {
                'left_wins': left_wins,
                'right_wins': right_wins,
                'skips': skips
            }
        except Exception:
            pass
        report_path = os.path.join(output_dir, 'report.json')
        with open(report_path, 'w', encoding='utf-8') as f:
            json.dump(convert_numpy_types(report_output), f, ensure_ascii=False, indent=4)
        logging.info(f"Saved report to {report_path}")
    except Exception as e:
        logging.error(f"Error saving report JSON: {str(e)}\n{traceback.format_exc()}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fencer Analysis Script.")
    parser.add_argument("--analysis_dir", type=str, default="./result/match_analysis", help="Directory with match analysis JSON files.")
    parser.add_argument("--match_data_dir", type=str, default="./result/match_data", help="Directory with match data CSV files.")
    parser.add_argument("--output_dir", type=str, default="./result/fencer_analysis", help="Directory to save analysis results.")
    parser.add_argument("--fps", type=int, default=30, help="Frames per second of the video.")
    parser.add_argument("--show-dfs", action="store_true", help="Print head of loaded dataframes for debugging.")

    args = parser.parse_args()

    main(
        analysis_dir=args.analysis_dir,
        match_data_dir=args.match_data_dir,
        output_dir=args.output_dir,
        fps=args.fps,
        show_dfs=args.show_dfs
    )
