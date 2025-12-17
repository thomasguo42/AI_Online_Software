import logging
from typing import Dict, List, Tuple, Any

def classify_fencer_touch_category(fencer_data: Dict, total_frames: int) -> str:
    """
    Classify a fencer's touch category based on their movement patterns.
    
    Args:
        fencer_data: Dictionary containing fencer's movement data
        total_frames: Total number of frames in the bout
    
    Returns:
        str: 'in_box', 'attack', or 'defense'
    """
    # Rule 1: If video is less than 60 frames, it's in_box
    if total_frames < 60:
        return 'in_box'
    
    # Get movement intervals
    advance_intervals = fencer_data.get('advance', [])
    pause_intervals = fencer_data.get('pause', [])
    retreat_intervals = fencer_data.get('retreat_intervals', [])
    
    # Combine pause and retreat intervals as retreat
    all_retreat_intervals = pause_intervals + retreat_intervals
    
    # Calculate total frames for each type
    advance_frames = sum(end - start + 1 for start, end in advance_intervals)
    retreat_frames = sum(end - start + 1 for start, end in all_retreat_intervals)
    
    # Rule 2: Check if majority is retreat
    total_movement_frames = advance_frames + retreat_frames
    if total_movement_frames > 0:
        retreat_ratio = retreat_frames / total_movement_frames
        if retreat_ratio > 0.5:
            return 'defense'
    
    # Rule 3: Check if last interval is retreat
    if all_retreat_intervals:
        # Find the last interval (highest end frame)
        last_retreat_end = max(end for start, end in all_retreat_intervals)
        last_advance_end = max((end for start, end in advance_intervals), default=-1)
        
        if last_retreat_end > last_advance_end:
            return 'defense'
    
    # Default to attack if not retreat
    return 'attack'

def _total_interval_frames(intervals: List[Tuple[int, int]]) -> int:
    """Safely sum total frames covered by a list of (start, end) frame intervals."""
    try:
        return sum((int(end) - int(start) + 1) for start, end in intervals if isinstance(start, int) and isinstance(end, int))
    except Exception:
        total = 0
        for interval in intervals:
            try:
                start, end = interval
                total += int(end) - int(start) + 1
            except Exception:
                continue
        return total

def classify_bout_categories(left_data: Dict, right_data: Dict, total_frames: int, fps: int) -> Tuple[str, str]:
    """
    Classify both fencers' categories for a bout using pairwise retreat comparison.

    Rule:
    - If bout duration < 60 frames => both 'in_box'.
    - Else, the fencer with the larger portion of retreat interval is 'defense',
      and the other fencer is 'attack'.

    Args:
        left_data: Left fencer data dict
        right_data: Right fencer data dict
        total_frames: Frames in the bout (end - start + 1)
        fps: Frames per second of the video

    Returns:
        (left_category, right_category)
    """
    # In-Box cutoff by frames
    if total_frames is None or int(total_frames) < 60:
        return 'in_box', 'in_box'

    # Gather retreat intervals (support both legacy top-level keys and nested movement_data)
    left_md = left_data.get('movement_data', {}) or {}
    right_md = right_data.get('movement_data', {}) or {}
    left_retreat = (left_data.get('retreat_intervals') or left_md.get('retreat_intervals') or [])
    right_retreat = (right_data.get('retreat_intervals') or right_md.get('retreat_intervals') or [])

    left_retreat_frames = _total_interval_frames(left_retreat)
    right_retreat_frames = _total_interval_frames(right_retreat)

    # Primary decision by absolute retreat frames (same denominator across fencers)
    if left_retreat_frames > right_retreat_frames:
        return 'defense', 'attack'
    if right_retreat_frames > left_retreat_frames:
        return 'attack', 'defense'

    # Tie-breaker 1: Compare retreat ratios relative to each fencer's movement frames
    left_advance = (left_data.get('advance') or left_md.get('advance_intervals') or [])
    right_advance = (right_data.get('advance') or right_md.get('advance_intervals') or [])
    left_movement_frames = _total_interval_frames(left_retreat) + _total_interval_frames(left_advance)
    right_movement_frames = _total_interval_frames(right_retreat) + _total_interval_frames(right_advance)

    left_ratio = (left_retreat_frames / left_movement_frames) if left_movement_frames > 0 else 0.0
    right_ratio = (right_retreat_frames / right_movement_frames) if right_movement_frames > 0 else 0.0

    if left_ratio > right_ratio:
        return 'defense', 'attack'
    if right_ratio > left_ratio:
        return 'attack', 'defense'

    # Tie-breaker 2: Who retreated latest (last retreat interval end)
    left_last_retreat_end = max((end for start, end in left_retreat), default=-1)
    right_last_retreat_end = max((end for start, end in right_retreat), default=-1)
    if left_last_retreat_end > right_last_retreat_end:
        return 'defense', 'attack'
    if right_last_retreat_end > left_last_retreat_end:
        return 'attack', 'defense'

    # Final fallback: both attack (no discernible defensive retreat advantage)
    return 'attack', 'attack'

def classify_bout_touches(bout_data: Dict) -> Dict[str, Any]:
    """
    Classify both fencers' touch categories for a bout.
    
    Args:
        bout_data: Dictionary containing bout analysis data
    
    Returns:
        Dictionary with classification results
    """
    total_frames = bout_data['frame_range'][1] - bout_data['frame_range'][0] + 1
    fps = bout_data.get('fps', 30)
    
    # Use pairwise classification based on retreat portions and duration (seconds)
    left_category, right_category = classify_bout_categories(
        bout_data['left_data'],
        bout_data['right_data'],
        total_frames,
        fps
    )
    
    # Get bout result from multiple possible sources
    bout_result = bout_data.get('bout_result', 'undetermined')
    winner_side = bout_data.get('winner_side', 'undetermined')
    
    # Also check judgement field which contains AI-determined winner
    judgement = bout_data.get('judgement', {})
    judgement_winner = judgement.get('winner', 'undetermined') if isinstance(judgement, dict) else 'undetermined'
    
    # Use winner_side if available, then judgement, then bout_result
    if winner_side and winner_side != 'undetermined':
        actual_winner = winner_side
    elif judgement_winner and judgement_winner not in ['undetermined', 'skip']:
        actual_winner = judgement_winner
    elif bout_result and bout_result not in ['skip', 'undetermined']:
        actual_winner = bout_result
    else:
        actual_winner = 'undetermined'
    
    return {
        'match_idx': bout_data['match_idx'],
        'total_frames': total_frames,
        'left_category': left_category,
        'right_category': right_category,
        'winner': actual_winner,
        'left_won': actual_winner == 'left',
        'right_won': actual_winner == 'right'
    }

def aggregate_touch_statistics(bout_classifications: List[Dict]) -> Dict[str, Any]:
    """
    Aggregate touch category statistics for both fencers.
    
    Args:
        bout_classifications: List of bout classification results
    
    Returns:
        Dictionary with aggregated statistics
    """
    stats = {
        'left_fencer': {
            'total_bouts': 0,
            'in_box': {'count': 0, 'wins': 0, 'losses': 0},
            'attack': {'count': 0, 'wins': 0, 'losses': 0},
            'defense': {'count': 0, 'wins': 0, 'losses': 0}
        },
        'right_fencer': {
            'total_bouts': 0,
            'in_box': {'count': 0, 'wins': 0, 'losses': 0},
            'attack': {'count': 0, 'wins': 0, 'losses': 0},
            'defense': {'count': 0, 'wins': 0, 'losses': 0}
        }
    }
    
    for bout in bout_classifications:
        # Left fencer stats
        stats['left_fencer']['total_bouts'] += 1
        left_cat = bout['left_category']
        stats['left_fencer'][left_cat]['count'] += 1
        
        if bout['left_won']:
            stats['left_fencer'][left_cat]['wins'] += 1
        elif bout['winner'] != 'undetermined':
            stats['left_fencer'][left_cat]['losses'] += 1
        
        # Right fencer stats
        stats['right_fencer']['total_bouts'] += 1
        right_cat = bout['right_category']
        stats['right_fencer'][right_cat]['count'] += 1
        
        if bout['right_won']:
            stats['right_fencer'][right_cat]['wins'] += 1
        elif bout['winner'] != 'undetermined':
            stats['right_fencer'][right_cat]['losses'] += 1
    
    # Calculate win rates
    for fencer in ['left_fencer', 'right_fencer']:
        for category in ['in_box', 'attack', 'defense']:
            cat_data = stats[fencer][category]
            total_decided = cat_data['wins'] + cat_data['losses']
            cat_data['win_rate'] = (cat_data['wins'] / total_decided * 100) if total_decided > 0 else 0
    
    return stats

def classify_all_bouts(bout_data_list: List[Dict]) -> Tuple[List[Dict], Dict[str, Any]]:
    """
    Classify all bouts and generate statistics.
    
    Args:
        bout_data_list: List of bout analysis dictionaries
    
    Returns:
        Tuple of (bout_classifications, aggregated_stats)
    """
    bout_classifications = []
    
    for bout_data in bout_data_list:
        try:
            classification = classify_bout_touches(bout_data)
            bout_classifications.append(classification)
            logging.debug(f"Bout {classification['match_idx']}: "
                         f"Left={classification['left_category']}, "
                         f"Right={classification['right_category']}, "
                         f"Winner={classification['winner']}")
        except Exception as e:
            logging.error(f"Error classifying bout {bout_data.get('match_idx', 'unknown')}: {e}")
            continue
    
    aggregated_stats = aggregate_touch_statistics(bout_classifications)
    
    logging.info(f"Classified {len(bout_classifications)} bouts successfully")
    return bout_classifications, aggregated_stats