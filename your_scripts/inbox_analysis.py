import logging
import json
import os
from typing import Dict, List, Any, Optional, Tuple
import numpy as np

def classify_attack_type_from_intervals(interval_analysis: Dict, has_launch: bool, arm_extensions: List) -> Tuple[str, float, List[str]]:
    """
    Classify attack type based on interval analysis data.
    
    Returns:
        Tuple of (attack_type_label, confidence_score, features_used)
    """
    if not interval_analysis or 'advance_analyses' not in interval_analysis:
        return 'no_attack', 1.0, ['no_advance_intervals']
    
    advance_analyses = interval_analysis['advance_analyses']
    if not advance_analyses:
        return 'no_attack', 1.0, ['no_advance_intervals']
    
    # Analyze attack patterns from advance intervals
    attack_types = []
    features_used = []
    
    for advance in advance_analyses:
        attack_info = advance.get('attack_info', {})
        attack_type = attack_info.get('attack_type', 'no_attack')
        num_extensions = attack_info.get('num_extensions', 0)
        num_launches = attack_info.get('num_launches', 0)
        tempo_type = advance.get('tempo_type', 'steady_tempo')
        tempo_changes = advance.get('tempo_changes', 0)
        
        attack_types.append(attack_type)
        
        if attack_type != 'no_attack':
            features_used.extend([
                f"attack_type_{attack_type}",
                f"extensions_{num_extensions}",
                f"launches_{num_launches}",
                f"tempo_{tempo_type}",
                f"tempo_changes_{tempo_changes}"
            ])
    
    # Determine overall attack classification
    if not any(at != 'no_attack' for at in attack_types):
        return 'no_attack', 1.0, ['no_attacking_actions']
    
    # Count attack patterns
    simple_attacks = attack_types.count('simple_attack')
    simple_preps = attack_types.count('simple_preparation')
    
    # Classification logic based on existing attack_info
    if simple_attacks > 0 and has_launch:
        if len(arm_extensions) > 0:
            # Check if arm extension comes early (within first 0.5s)
            early_extension = any(ext.get('start_frame', float('inf')) / 30 < 0.5 for ext in arm_extensions)
            if early_extension:
                return 'direct_lunge', 0.8, features_used + ['early_arm_extension', 'has_launch']
            else:
                return 'step_lunge', 0.7, features_used + ['late_arm_extension', 'has_launch']
        else:
            return 'direct_lunge', 0.6, features_used + ['has_launch', 'no_arm_extension']
    
    elif simple_preps > 0:
        if len(arm_extensions) > 1:
            return 'feint_attack', 0.7, features_used + ['multiple_extensions', 'preparation']
        else:
            return 'short_burst', 0.6, features_used + ['single_preparation']
    
    elif simple_attacks > 0:
        return 'simple_attack', 0.8, features_used + ['simple_attack_detected']
    
    else:
        return 'complex_action', 0.5, features_used + ['mixed_attack_types']

def extract_inbox_bout_details(bout_data: Dict, fps: int) -> Optional[Dict]:
    """
    Extract detailed In-Box analysis from a single bout.
    
    Args:
        bout_data: Single bout data from match_analysis.json
        fps: Frames per second
    
    Returns:
        Dictionary with In-Box analysis for both fencers, or None if not In-Box
    """
    frame_range = bout_data.get('frame_range', [0, 60])
    total_frames = frame_range[1] - frame_range[0] + 1
    
    # Check if this is an In-Box bout (< 60 frames)
    if total_frames >= 60:
        return None
    
    bout_duration_s = total_frames / fps
    
    result = {
        'meta': {
            'fps': fps,
            'total_frames': total_frames,
            'bout_duration_s': bout_duration_s,
            'frame_range': frame_range
        },
        'left_fencer': {},
        'right_fencer': {}
    }
    
    # Process both fencers
    for side in ['left', 'right']:
        fencer_key = f'{side}_fencer' if side == 'left' else 'right_fencer'
        data_key = f'{side}_data'
        
        if data_key not in bout_data:
            continue
            
        fencer_data = bout_data[data_key]
        
        # Extract velocity metrics
        velocity_stats = {
            'mean': fencer_data.get('velocity', 0.0),
            'max': 0.0,  # Will calculate from movement data if available
            'p95': 0.0,  # Will calculate from movement data if available
            'forward_mean': 0.0,  # Could be derived from advance intervals
            'backward_mean': 0.0,  # Could be derived from retreat intervals
            'variability': 0.0
        }
        
        # Extract acceleration metrics
        acceleration_stats = {
            'mean': fencer_data.get('acceleration', 0.0),
            'max': 0.0,
            'p95': 0.0,
            'forward_mean': 0.0,
            'backward_mean': 0.0,
            'burst_count': 0
        }
        
        # Extract pause information
        pause_intervals = fencer_data.get('pause_sec', [])
        total_pause_duration = sum([(end - start) for start, end in pause_intervals])
        pause_stats = {
            'present': len(pause_intervals) > 0,
            'total_seconds': total_pause_duration,
            'share': total_pause_duration / bout_duration_s if bout_duration_s > 0 else 0,
            'count': len(pause_intervals),
            'last_end_seconds': fencer_data.get('latest_pause_retreat_end', -1) / fps if fencer_data.get('latest_pause_retreat_end', -1) != -1 else None
        }
        
        # Extract lunge information
        launches = fencer_data.get('launches', [])
        has_launch = fencer_data.get('has_launch', False)
        launch_stats = {
            'present': has_launch,
            'launch_time_s': None,
            'peak_velocity': 0.0,
            'peak_acceleration': 0.0,
            'distance': 0.0,
            'promptness_s': None
        }
        
        if has_launch and launches:
            first_launch = launches[0]
            launch_stats.update({
                'launch_time_s': first_launch.get('start_frame', 0) / fps,
                'peak_velocity': first_launch.get('front_foot_max_velocity', 0.0),
                'peak_acceleration': first_launch.get('front_foot_max_acceleration', 0.0),
                'distance': first_launch.get('max_foot_distance', 0.0) - first_launch.get('initial_foot_distance', 0.0),
                'promptness_s': first_launch.get('promptness_ratio', 0.0) * bout_duration_s
            })
        
        # Classify attack type
        interval_analysis = fencer_data.get('interval_analysis', {})
        extensions = fencer_data.get('extensions', [])
        attack_type_label, confidence, features = classify_attack_type_from_intervals(
            interval_analysis, has_launch, extensions
        )
        
        attack_type_stats = {
            'label': attack_type_label,
            'confidence': confidence,
            'features_used': features
        }
        
        # Extract arm holding information
        arm_extensions = fencer_data.get('extensions', [])
        extension_frames = sum([ext.get('duration_frames', 0) for ext in arm_extensions])
        holding_arm_stats = {
            'ratio': extension_frames / total_frames if total_frames > 0 else 0,
            'avg_angle': 0.0,  # Would need to calculate from extension data
            'early_arm': False,
            'dwell_seconds': extension_frames / fps
        }
        
        # Check for early arm extension (within first 0.5 seconds)
        if arm_extensions:
            first_extension_start = arm_extensions[0].get('start_frame', float('inf')) / fps
            holding_arm_stats['early_arm'] = first_extension_start < 0.5
            
            # Calculate average arm angle
            angles = []
            for ext in arm_extensions:
                if 'arm_angle_profile' in ext:
                    angles.extend(ext['arm_angle_profile'])
            holding_arm_stats['avg_angle'] = np.mean(angles) if angles else 0.0
        
        # Extract initial step information
        first_step = fencer_data.get('first_step', {})
        initial_step_stats = {
            'onset_time_s': first_step.get('init_time', 0.0),
            'initial_velocity': first_step.get('velocity', 0.0),
            'time_to_peak_s': 0.0,  # Would need to calculate from velocity profile
            'amplitude': first_step.get('momentum', 0.0)  # Using momentum as proxy for amplitude
        }
        
        # Compile all stats for this fencer
        result[fencer_key] = {
            'velocity': velocity_stats,
            'acceleration': acceleration_stats,
            'pause': pause_stats,
            'lunge': launch_stats,
            'attack_type': attack_type_stats,
            'holding_arm': holding_arm_stats,
            'initial_step': initial_step_stats
        }
    
    return result

def _load_json_safely(file_path: str) -> Optional[Dict[str, Any]]:
    """Load JSON and sanitize invalid numeric tokens like Infinity/NaN."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            raw = f.read()
        # Replace invalid JSON tokens
        sanitized = (
            raw.replace('Infinity', 'null')
               .replace('-Infinity', 'null')
               .replace('NaN', 'null')
        )
        return json.loads(sanitized)
    except Exception as e:
        logging.error(f"Failed to load JSON {file_path}: {e}")
        return None


def process_inbox_bouts_from_analysis(analysis_dir: str) -> Dict[str, Any]:
    """
    Process all match analysis files and extract In-Box bout details.
    
    Args:
        analysis_dir: Directory containing match_analysis JSON files
    
    Returns:
        Dictionary with processed In-Box data
    """
    inbox_bouts = []
    
    if not os.path.exists(analysis_dir):
        logging.warning(f"Analysis directory not found: {analysis_dir}")
        return {'bouts': [], 'summary': {}}
    
    # Process all match analysis files
    for filename in os.listdir(analysis_dir):
        if not filename.endswith('_analysis.json'):
            continue
            
        filepath = os.path.join(analysis_dir, filename)
        
        try:
            bout_data = _load_json_safely(filepath)
            if bout_data is None:
                continue
            
            fps = bout_data.get('fps', 30)
            inbox_details = extract_inbox_bout_details(bout_data, fps)
            
            if inbox_details:
                inbox_details['match_idx'] = bout_data.get('match_idx')
                inbox_details['upload_id'] = bout_data.get('upload_id')
                inbox_details['filename'] = filename
                inbox_bouts.append(inbox_details)
                
        except Exception as e:
            logging.error(f"Error processing {filepath}: {e}")
            continue
    
    # Generate summary statistics
    summary = generate_inbox_summary(inbox_bouts)
    
    return {
        'bouts': inbox_bouts,
        'summary': summary,
        'total_inbox_bouts': len(inbox_bouts)
    }

def generate_inbox_summary(inbox_bouts: List[Dict]) -> Dict[str, Any]:
    """
    Generate summary statistics for In-Box bouts.
    
    Args:
        inbox_bouts: List of processed In-Box bout data
    
    Returns:
        Summary statistics dictionary
    """
    if not inbox_bouts:
        return {}
    
    left_stats = []
    right_stats = []
    
    for bout in inbox_bouts:
        left_stats.append(bout.get('left_fencer', {}))
        right_stats.append(bout.get('right_fencer', {}))
    
    def summarize_fencer_stats(fencer_stats_list: List[Dict]) -> Dict:
        """Summarize statistics for one fencer across all bouts."""
        if not fencer_stats_list:
            return {}
        
        # Collect metrics
        velocities = [fs.get('velocity', {}).get('mean', 0) for fs in fencer_stats_list]
        accelerations = [fs.get('acceleration', {}).get('mean', 0) for fs in fencer_stats_list]
        attack_types = [fs.get('attack_type', {}).get('label', 'no_attack') for fs in fencer_stats_list]
        lunge_counts = sum([1 for fs in fencer_stats_list if fs.get('lunge', {}).get('present', False)])
        early_arm_counts = sum([1 for fs in fencer_stats_list if fs.get('holding_arm', {}).get('early_arm', False)])
        
        return {
            'total_bouts': len(fencer_stats_list),
            'avg_velocity': np.mean(velocities) if velocities else 0,
            'avg_acceleration': np.mean(accelerations) if accelerations else 0,
            'attack_type_distribution': {
                atype: attack_types.count(atype) for atype in set(attack_types)
            },
            'lunge_rate': lunge_counts / len(fencer_stats_list) if fencer_stats_list else 0,
            'early_arm_rate': early_arm_counts / len(fencer_stats_list) if fencer_stats_list else 0
        }
    
    return {
        'left_fencer': summarize_fencer_stats(left_stats),
        'right_fencer': summarize_fencer_stats(right_stats),
        'total_bouts_analyzed': len(inbox_bouts)
    }

def integrate_inbox_with_touch_classification(bout_classifications: List[Dict], 
                                            bout_data: List[Dict], 
                                            fps: int) -> Dict[str, Any]:
    """
    Integrate In-Box analysis with touch classification results.
    
    Args:
        bout_classifications: List of bout classifications with touch categories
        bout_data: List of bout data from match analysis
        fps: Frames per second
    
    Returns:
        Enhanced touch classification with In-Box details
    """
    inbox_enhanced_classifications = []
    
    for i, classification in enumerate(bout_classifications):
        enhanced = classification.copy()
        
        # Check if this bout is In-Box for both fencers
        if (classification.get('left_category') == 'in_box' and 
            classification.get('right_category') == 'in_box'):
            
            # Find corresponding bout data
            match_idx = classification.get('match_idx')
            corresponding_bout = None
            
            for bout in bout_data:
                if bout.get('match_idx') == match_idx:
                    corresponding_bout = bout
                    break
            
            if corresponding_bout:
                inbox_details = extract_inbox_bout_details(corresponding_bout, fps)
                if inbox_details:
                    enhanced['inbox_details'] = inbox_details
        
        inbox_enhanced_classifications.append(enhanced)
    
    return {
        'classifications': inbox_enhanced_classifications,
        'inbox_count': sum([1 for c in inbox_enhanced_classifications if 'inbox_details' in c])
    }