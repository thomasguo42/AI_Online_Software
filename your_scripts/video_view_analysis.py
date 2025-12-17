#!/usr/bin/env python3
"""
Video View Analysis - Performance metrics calculation for saber fencing analysis
Implements the detailed formulas for 9 performance metrics across In-Box, Attack, and Defense categories
"""

import base64
import io
import os
import json
import logging
import math
import re
import random
import requests
import time
from collections import defaultdict
from statistics import mean
from typing import Dict, List, Tuple, Optional, Any

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial', 'sans-serif']
plt.rcParams['axes.unicode_minus'] = False

from your_scripts.bout_classification import classify_bout_categories

_RESULTS_FOLDER_ENV = os.getenv('RESULT_FOLDER', '').strip()
_RESULTS_FOLDER_ROOT = (
    _RESULTS_FOLDER_ENV
    if _RESULTS_FOLDER_ENV and os.path.isabs(_RESULTS_FOLDER_ENV)
    else os.path.join(os.getcwd(), _RESULTS_FOLDER_ENV or 'results')
)
_DISPLAY_VIDEO_CACHE: Dict[Tuple[int, int, int], str] = {}


def _get_display_video_relpath(user_id: int, upload_id: int, match_idx: int) -> str:
    """Return relative video path with extended padding when available."""
    cache_key = (int(user_id), int(upload_id), int(match_idx))
    if cache_key in _DISPLAY_VIDEO_CACHE:
        return _DISPLAY_VIDEO_CACHE[cache_key]

    user_part = str(user_id)
    upload_part = str(upload_id)
    match_folder = f'match_{match_idx}'
    extended_filename = f'{match_folder}_extended.mp4'
    regular_filename = f'{match_folder}.mp4'

    extended_abs = os.path.join(
        _RESULTS_FOLDER_ROOT, user_part, upload_part, 'matches', match_folder, extended_filename
    )
    if os.path.exists(extended_abs):
        rel_path = '/'.join([user_part, upload_part, 'matches', match_folder, extended_filename])
        _DISPLAY_VIDEO_CACHE[cache_key] = rel_path
        return rel_path

    regular_abs = os.path.join(
        _RESULTS_FOLDER_ROOT, user_part, upload_part, 'matches', match_folder, regular_filename
    )
    if os.path.exists(regular_abs):
        rel_path = '/'.join([user_part, upload_part, 'matches', match_folder, regular_filename])
        _DISPLAY_VIDEO_CACHE[cache_key] = rel_path
        return rel_path

    _DISPLAY_VIDEO_CACHE[cache_key] = ''
    return ''

CATEGORY_LABELS = {
    'in_box': 'Inbox',
    'attack': 'Attack',
    'defense': 'Defense'
}

CATEGORY_DEFINITIONS = {
    'in_box': 'Both fencers start simultaneously or enter close-range engagement, measuring efficiency in simultaneous attack scenarios.',
    'attack': 'Our fencer initiates first, controlling the right-of-way, recording scoring efficiency when actively attacking.',
    'defense': 'Opponent attacks first, our fencer attempts to score through retreat, parry, or counter-attack, reflecting defensive system quality.'
}

SIDE_LABELS = {'left': 'Left Fencer', 'right': 'Right Fencer'}

LOSS_REASON_TRANSLATIONS = {
    'Slow Reaction at Start': 'Slow Reaction at Start',
    'Outmatched by Speed & Power': 'Outmatched by Speed & Power',
    'Indecisive Movement / Early Pause': 'Indecisive Movement / Early Pause',
    'Lack of Offensive Commitment': 'Lack of Offensive Commitment',
    'Lack of Arm Extension': 'Lack of Arm Extension',
    'Lack of Lunging': 'Lack of Lunging',
    'Attacked from Too Far (Positional Error)': 'Attacked from Too Far (Positional Error)',
    'Predictable Attack (Tactical Error)': 'Predictable Attack (Tactical Error)',
    'Countered on Preparation (Timing Error)': 'Countered on Preparation (Timing Error)',
    'Passive/Weak Attack (Execution Failure)': 'Passive/Weak Attack (Execution Failure)',
    'Collapsed Distance (Positional Error)': 'Collapsed Distance (Positional Error)',
    'Failed to "Pull Distance" vs. Lunge (Positional Error)': 'Failed to "Pull Distance" vs. Lunge (Positional Error)',
    'Missed Counter-Attack Opportunity (Tactical Error)': 'Missed Counter-Attack Opportunity (Tactical Error)',
    'Purely Defensive / No Counter-Threat (Execution Failure)': 'Purely Defensive / No Counter-Threat (Execution Failure)',
    'General/Unclassified': 'General/Unclassified'
}

WIN_REASON_TRANSLATIONS = {
    'Superior Reaction at Start': 'Superior Reaction at Start',
    'Overpowering at Start': 'Overpowering at Start',
    'Exploited Hesitation': 'Exploited Hesitation',
    'Superior Positioning - Optimal Distance': 'Superior Positioning - Optimal Distance',
    'Superior Tactics - Rhythm Break': 'Superior Tactics - Rhythm Break',
    'Superior Power - Overwhelming Attack': 'Superior Power - Overwhelming Attack',
    'Capitalized on Attacker Positional Error': 'Capitalized on Attacker Positional Error',
    'Capitalized on Attacker Rhythmic Error': 'Capitalized on Attacker Rhythmic Error',
    'Capitalized on Attacker Power Failure': 'Capitalized on Attacker Power Failure',
    'General/Unclassified': 'General/Unclassified'
}

# Build case-insensitive lookup maps for reason normalization
_LOSS_REASON_LC_MAP = {str(k).lower(): v for k, v in LOSS_REASON_TRANSLATIONS.items()}
_WIN_REASON_LC_MAP = {str(k).lower(): v for k, v in WIN_REASON_TRANSLATIONS.items()}


def _normalize_reason_text(label: Any) -> str:
    """Strip numeric/prefix noise and return a cleaned reason label for grouping/display."""
    try:
        text = str(label)
    except Exception:
        text = ''
    if not text:
        return ''
    # Remove bracketed prefixes like "[Attack]" if present
    cleaned = _strip_category_prefix(text)
    # Remove leading 'Sub-category' or 'Subcategory' (case-insensitive)
    cleaned = re.sub(r'^\s*sub[-\s]*category\s*', '', cleaned, flags=re.IGNORECASE)
    # Remove leading numeric prefixes like '1.2:', '1:', '1)'
    cleaned = re.sub(r'^\s*\d+(?:\.\d+)?\s*[:\).]\s*', '', cleaned)
    # Collapse whitespace and trim trailing punctuation
    cleaned = re.sub(r'\s+', ' ', cleaned).strip(' -:\u3000').strip()
    return cleaned


def _canonical_reason_label(raw_label: Any) -> str:
    """Map various label variants to a canonical display form using known dictionaries; fallback to cleaned text."""
    base = _normalize_reason_text(raw_label)
    lc = base.lower()
    # Prefer explicit mappings to ensure consistent casing
    if lc in _LOSS_REASON_LC_MAP:
        return _LOSS_REASON_LC_MAP[lc]
    if lc in _WIN_REASON_LC_MAP:
        return _WIN_REASON_LC_MAP[lc]
    return base

GEMINI_MODEL = os.getenv('GEMINI_MODEL', 'gemini-2.5-flash-lite')

# Centralized API pacing configuration to reduce 429/503 responses
GEMINI_MIN_INTERVAL_S = float(os.getenv('GEMINI_MIN_INTERVAL_S', '2.5'))
GEMINI_BACKOFF_BASE_S = float(os.getenv('GEMINI_BACKOFF_BASE_S', '2.0'))
GEMINI_BACKOFF_MAX_S = float(os.getenv('GEMINI_BACKOFF_MAX_S', '12'))
GEMINI_TOUCH_DELAY_S = float(os.getenv('GEMINI_TOUCH_DELAY_S', '1.5'))

# Global pacing cursor shared across all Gemini calls in this process
_GEMINI_NEXT_AVAILABLE_TS: float = 0.0


# === Heuristic thresholds for deterministic in-box classification ===
INBOX_REACTION_THRESHOLD = 0.045  # seconds advantage
INBOX_SECONDARY_REACTION_THRESHOLD = 0.025  # fallback threshold when no other cue
INBOX_DOMINANCE_VELOCITY_DIFF = 0.35  # m/s difference
INBOX_DOMINANCE_ACCEL_DIFF = 1.2      # m/s^2 difference
INBOX_HESITATION_START = 0.6          # seconds, early hesitation window
INBOX_HESITATION_MIN_DURATION = 0.12  # seconds, meaningful pause duration


def _safe_float(value) -> Optional[float]:
    """Convert to float if numeric and finite, otherwise return None."""
    if isinstance(value, (int, float)) and not math.isnan(value) and not math.isinf(value):
        return float(value)
    return None


def _format_seconds(value: Optional[float], precision: int = 3) -> str:
    """Format seconds with fixed precision for human-readable evidence."""
    if value is None:
        return 'No data'
    return f"{value:.{precision}f}s"


def _format_number(value: Optional[float], precision: int = 2) -> str:
    if value is None:
        return 'No data'
    return f"{value:.{precision}f}"


def _first_pause_window(fencer_data: Dict) -> Optional[Tuple[float, float]]:
    """Return earliest pause (start, duration) in seconds if available."""
    pause_sec = fencer_data.get('pause_sec') or []
    candidate: Optional[Tuple[float, float]] = None
    for interval in pause_sec:
        try:
            start = float(interval[0])
            end = float(interval[1]) if len(interval) > 1 else float(interval[0])
        except (TypeError, ValueError, IndexError):
            continue
        duration = max(0.0, end - start)
        if candidate is None or start < candidate[0]:
            candidate = (start, duration)
    return candidate


def _has_early_hesitation(fencer_data: Dict, ratio_threshold: float = 0.45) -> bool:
    """Determine whether the fencer shows early hesitation in in-box phase."""
    first_pause = _first_pause_window(fencer_data)
    if first_pause:
        start, duration = first_pause
        if start <= INBOX_HESITATION_START and duration >= INBOX_HESITATION_MIN_DURATION:
            return True
    pause_ratio = _safe_float(fencer_data.get('pause_ratio'))
    if pause_ratio is not None and pause_ratio >= ratio_threshold:
        return True
    return False


def _build_inbox_win_reason(touch_data: Dict, winner_side: str) -> Dict:
    """Deterministically classify in-box winning reason with evidence."""
    loser_side = 'right' if winner_side == 'left' else 'left'
    winner_data = touch_data.get(f'{winner_side}_data', {}) or {}
    loser_data = touch_data.get(f'{loser_side}_data', {}) or {}

    winner_step = winner_data.get('first_step', {}) or {}
    loser_step = loser_data.get('first_step', {}) or {}

    winner_init = _safe_float(winner_step.get('init_time'))
    loser_init = _safe_float(loser_step.get('init_time'))
    reaction_advantage = None
    if winner_init is not None and loser_init is not None:
        reaction_advantage = loser_init - winner_init  # positive => winner earlier

    winner_velocity = _safe_float(winner_step.get('velocity'))
    loser_velocity = _safe_float(loser_step.get('velocity'))
    velocity_diff = None
    if winner_velocity is not None and loser_velocity is not None:
        velocity_diff = winner_velocity - loser_velocity

    winner_acc = _safe_float(winner_step.get('acceleration'))
    loser_acc = _safe_float(loser_step.get('acceleration'))
    acceleration_diff = None
    if winner_acc is not None and loser_acc is not None:
        acceleration_diff = winner_acc - loser_acc

    loser_hesitation = _has_early_hesitation(loser_data)
    winner_hesitation = _has_early_hesitation(winner_data, ratio_threshold=0.6)

    reason_key = None
    rationale_lines: List[str] = []

    # Prioritise hesitation cues when they are unilateral
    if loser_hesitation and not winner_hesitation:
        reason_key = 'Exploited Hesitation'
        rationale_lines.append('Opponent showed pause/deceleration early after command, you maintained advance to score.')
    elif reaction_advantage is not None and reaction_advantage >= INBOX_REACTION_THRESHOLD:
        reason_key = 'Superior Reaction at Start'
        rationale_lines.append('You started first after command, clearly ahead of opponent entering engagement.')
    elif (velocity_diff is not None and velocity_diff >= INBOX_DOMINANCE_VELOCITY_DIFF) or \
         (acceleration_diff is not None and acceleration_diff >= INBOX_DOMINANCE_ACCEL_DIFF):
        reason_key = 'Overpowering at Start'
        rationale_lines.append('Both started almost simultaneously, but your first step speed/power far exceeded opponent.')
    elif loser_hesitation:
        # Both hesitated but opponent more severe
        reason_key = 'Exploited Hesitation'
        rationale_lines.append('Similar reactions, but opponent decelerated quickly after start, you seized the opportunity.')
    else:
        # Fallback based on strongest available cue
        if reaction_advantage is not None and reaction_advantage >= INBOX_SECONDARY_REACTION_THRESHOLD:
            reason_key = 'Superior Reaction at Start'
            rationale_lines.append('Your starting reaction advantage maintained route advantage.')
        elif velocity_diff is not None and velocity_diff > 0:
            reason_key = 'Overpowering at Start'
            rationale_lines.append('Synchronized start, but advance intensity advantage on your side.')
        else:
            reason_key = 'Superior Reaction at Start'
            rationale_lines.append('Limited data, default determination is starting reaction advantage.')

    data_evidence = []
    if reaction_advantage is not None:
        data_evidence.append(
            f"Reaction difference: Opponent {_format_seconds(loser_init)} vs You {_format_seconds(winner_init)}"
        )
    if velocity_diff is not None:
        data_evidence.append(
            f"First step velocity: You {_format_number(winner_velocity)} vs Opponent {_format_number(loser_velocity)}"
        )
    if acceleration_diff is not None:
        data_evidence.append(
            f"First step acceleration diff: {_format_number(acceleration_diff)}"
        )

    supporting_actions = []
    loser_pause = _first_pause_window(loser_data)
    if loser_pause:
        start, duration = loser_pause
        supporting_actions.append(
            f"Opponent showed {_format_seconds(duration, precision=2)} pause at {_format_seconds(start)}"
        )
    if winner_hesitation:
        supporting_actions.append('You also had slight pause in early start, need to maintain continuous advance.')

    return {
        'win_category': 'In-Box',
        'win_sub_category': reason_key,
        'brief_reasoning': '；'.join(rationale_lines),
        'data_evidence': data_evidence[:4],
        'supporting_actions': supporting_actions[:3]
    }


def _extract_top_loss_patterns(loss_analysis: Dict, fencer_side: str, max_entries: int = 5) -> List[Dict]:
    patterns: List[Dict] = []
    if not loss_analysis:
        return patterns

    key = f'{fencer_side}_fencer'
    if key not in loss_analysis:
        return patterns

    for category, reasons in loss_analysis[key].items():
        for reason_key, info in reasons.items():
            count = info.get('count', 0) or 0
            if count <= 0:
                continue
            touches = info.get('touches', []) or []
            example_indices = []
            for touch in touches:
                idx = touch.get('touch_index')
                if isinstance(idx, int):
                    example_indices.append(idx + 1)
            patterns.append({
                'category': category,
                'reason_key': reason_key,
                'reason_cn': LOSS_REASON_TRANSLATIONS.get(reason_key, reason_key),
                'count': count,
                'examples': example_indices[:3],
                'raw': info
            })

    patterns.sort(key=lambda item: item['count'], reverse=True)
    return patterns[:max_entries]


def _extract_top_win_patterns(win_analysis: Dict, fencer_side: str, max_entries: int = 5) -> List[Dict]:
    patterns: List[Dict] = []
    if not win_analysis:
        return patterns

    key = f'{fencer_side}_fencer'
    if key not in win_analysis:
        return patterns

    for category, reasons in win_analysis[key].items():
        for reason_key, info in reasons.items():
            count = info.get('count', 0) or 0
            if count <= 0:
                continue
            touches = info.get('touches', []) or []
            example_indices = []
            for touch in touches:
                idx = touch.get('touch_index')
                if isinstance(idx, int):
                    example_indices.append(idx + 1)
            patterns.append({
                'category': category,
                'reason_key': reason_key,
                'reason_cn': WIN_REASON_TRANSLATIONS.get(reason_key, reason_key),
                'count': count,
                'examples': example_indices[:3],
                'raw': info
            })

    patterns.sort(key=lambda item: item['count'], reverse=True)
    return patterns[:max_entries]

LOSS_REASON_REMEDIATIONS = {
    'Slow Reaction at Start': [
        'Set up light/sound signal start drills (6 reps × 10 sets), target reaction time ≤0.18 seconds.',
        'Add "start within 0.3 seconds after referee command" training in engagement simulation, develop muscle memory for quick starts.',
        'Review match recordings frame-by-frame, analyze opponent pre-start signals to improve visual anticipation.'
    ],
    'Outmatched by Speed & Power': [
        'Arrange strength circuit: weighted lunge, squat, core rotation each 3 sets × 12 reps, improve lower body explosiveness.',
        'Add "acceleration phase" timing in attack drills, require last two steps to increase speed by 20%.',
        'Use resistance bands or weighted vests for attack drills, strengthen upper body speed and control.'
    ],
    'Indecisive Movement / Early Pause': [
        'Conduct rhythm control training: use metronome to guide advance rhythm, prohibit mid-action pauses.',
        'Set "no more than one consecutive retreat" rule in engagement simulation, force fencer to maintain initiative.',
        'Watch key bout videos, mark hesitation frames and conduct scenario reproduction training.'
    ],
    'Lack of Offensive Commitment': [
        'Establish "confirmed attack" process: clarify target area and follow-up actions before starting.',
        'Use "command-style" training: coach randomly calls out attack lines, require immediate execution of complete lunge.',
        'Set "penalty run for incomplete attack" rule in bouts, force fencer to complete full attack sequence.'
    ],
    'Attacked from Too Far (Positional Error)': [
        'Conduct distance sensitivity training: coach moves target back and forth, fencer attacks only after entering optimal distance.',
        'Use floor markers (0.5m intervals) in attack-defense drills to help establish distance reference.',
        'Analyze failed attack stride patterns from recordings, identify excessive step length and correct.'
    ],
    'Predictable Attack (Tactical Error)': [
        'Introduce feint module: each round must complete "probe-feint-real attack" three-phase sequence.',
        'Practice different rhythm combinations (fast-slow-fast, slow-fast-fast) to disrupt opponent defensive rhythm.',
        'Track habitual attack lines, force use of secondary lines ≥40% in training.'
    ],
    'Countered on Preparation (Timing Error)': [
        'Train preparation action concealment, reduce weapon hand premature extension time.',
        'Add "opponent counter-attack trigger" drill: excessive feint immediately countered, require line change or withdrawal.',
        'Observe own preparation action exposure points through mirror drills, correct systematically.'
    ],
    'Passive/Weak Attack (Execution Failure)': [
        'Conduct power-to-speed training: weighted weapon bag strikes, resistance band extension thrusts, each 3 sets × 15 reps.',
        'Set "attack must be two-tempo continuous" in bout drills, cultivate pursuit awareness.',
        'Use video feedback to check arm extension speed and footwork coordination, ensure lunge endpoint explosion.'
    ],
    'Collapsed Distance (Positional Error)': [
        'Train retreat rhythm: set "must stabilize after two retreats" drill, prevent continuous pressure.',
        'Add "maintain 0.6m safe distance" reminder in attack-defense transitions, strengthen spatial awareness.',
        'Simulate baseline pressure scenarios, practice lateral evasion or timely counter-pressure, avoid being pushed out.'
    ],
    'Failed to "Pull Distance" vs. Lunge (Positional Error)': [
        'Conduct retreat + counter-attack combination training: when opponent lunges, require distance pull first then counter.',
        'Strengthen pre-lunge prediction drills, identify shoulder and weapon hand preparation movements.',
        'Add "after pulling distance must immediately re-establish attack distance" component in training, avoid sustained pressure.'
    ],
    'Missed Counter-Attack Opportunity (Tactical Error)': [
        'Establish "trigger condition" checklist: clarify which actions are counter-attack signals and drill repeatedly.',
        'Arrange fast-paced engagement drills, require at least one active counter-attack per round.',
        'Analyze missed opportunity frames from recordings, identify hesitation points and conduct targeted simulation.'
    ],
    'Purely Defensive / No Counter-Threat (Execution Failure)': [
        'Set "must riposte after defense" scoring mechanism in defensive training, prevent passive defense.',
        'Train first step footwork after parry (advance one step or side step) to create counter-attack angle.',
        'Combine with conditioning training, ensure ability for explosive riposte after defense.'
    ],
    'General/Unclassified': [
        'Review key loss bouts, organize action chains and record specific stages leading to losses.',
        'Work with coach to create event-response strategy table, clarify handling plan for each loss type.',
        'Convert loss scenarios into required training items for next phase, form closed loop.'
    ]
}

def sanitize_value(value) -> float:
    """Sanitize a numeric value to ensure it's JSON serializable"""
    if value is None:
        return 0.0
    if isinstance(value, (int, float)):
        if math.isnan(value) or math.isinf(value):
            return 0.0
        return float(value)
    return 0.0

def normalize_value(value: float, min_val: float, max_val: float) -> float:
    """Normalize a value to 0-1 range"""
    value = sanitize_value(value)
    min_val = sanitize_value(min_val) 
    max_val = sanitize_value(max_val)
    
    if max_val == min_val:
        return 0.0
    return max(0.0, min(1.0, (value - min_val) / (max_val - min_val)))

def calculate_performance_metrics(upload_id: int, user_id: int) -> Dict:
    """
    Calculate comprehensive performance metrics for both fencers in an upload
    Returns data structure suitable for radar chart visualization
    """
    
    # Load all match analysis JSON files
    result_dir = f"results/{user_id}/{upload_id}"
    match_analysis_dir = os.path.join(result_dir, "match_analysis")
    
    if not os.path.exists(match_analysis_dir):
        raise FileNotFoundError(f"Match analysis directory not found: {match_analysis_dir}")
    
    # Load all match JSON files
    match_data = []
    for filename in os.listdir(match_analysis_dir):
        if filename.endswith('_analysis.json'):
            filepath = os.path.join(match_analysis_dir, filename)
            try:
                with open(filepath, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    # Inject filename and derived video path/match index for later use (loss analysis/video embedding)
                    data['filename'] = filename
                    match_idx = None
                    m = re.search(r'match_(\d+)_analysis\.json$', filename)
                    if m:
                        try:
                            match_idx = int(m.group(1))
                        except Exception:
                            match_idx = None
                    data['match_idx'] = match_idx
                    # Build relative video path under results folder (served by /results/<path:filename>)
                    if match_idx is not None:
                        data['video_path'] = _get_display_video_relpath(user_id, upload_id, match_idx)
                    else:
                        data['video_path'] = ''
                    data['touch_index'] = len(match_data)
                    match_data.append(data)
            except Exception as e:
                logging.error(f"Error loading {filepath}: {e}")
                continue
    
    if not match_data:
        raise ValueError("No match analysis files found")
    
    # Initialize counters for both fencers
    left_metrics = initialize_fencer_metrics()
    right_metrics = initialize_fencer_metrics()
    
    # Process each touch
    for touch_data in match_data:
        left_category = _determine_fencer_category(touch_data, 'left')
        right_category = _determine_fencer_category(touch_data, 'right')

        if touch_data.get('left_fencer_category') not in VALID_FENCER_CATEGORIES:
            touch_data['left_fencer_category'] = left_category
        if touch_data.get('right_fencer_category') not in VALID_FENCER_CATEGORIES:
            touch_data['right_fencer_category'] = right_category

        process_touch_for_metrics(touch_data, 'left', left_metrics)
        process_touch_for_metrics(touch_data, 'right', right_metrics)
    
    # Calculate final scores
    left_scores = calculate_final_scores(left_metrics)
    right_scores = calculate_final_scores(right_metrics)
    
    # Calculate bout type statistics
    bout_type_stats = calculate_bout_type_statistics(match_data)
    
    return {
        'left_fencer_metrics': left_scores,
        'right_fencer_metrics': right_scores,
        'bout_type_statistics': bout_type_stats,
        'total_touches': len(match_data)
    }

def initialize_fencer_metrics() -> Dict:
    """Initialize metric counters for a fencer"""
    return {
        # In-Box Metrics
        'first_intention_attempts': 0,
        'first_intention_wins': 0,
        'second_intention_attempts': 0,
        'second_intention_wins': 0,
        'attack_promptness_scores': [],
        
        # Attack Metrics
        'attack_aggressiveness_scores': [],
        'total_attacking_advances': 0,
        'good_distance_advances': 0,
        'attack_attempts': 0,
        'attack_wins': 0,
        
        # Defense Metrics
        'total_retreats': 0,
        'good_quality_retreats': 0,
        'total_counter_ops': 0,
        'counters_executed': 0,
        'defense_actions': 0,
        'defense_wins': 0,
        # Defense distance management via safe distance & spacing (defense-category only)
        'defense_total_retreat_intervals': 0,
        'defense_safe_distance_count': 0,
        'defense_consistent_spacing_count': 0
    }

def process_touch_for_metrics(touch_data: Dict, fencer_side: str, metrics: Dict):
    """Process a single touch to update metrics for specified fencer"""
    
    fencer_data = touch_data.get(f'{fencer_side}_data', {}) or {}
    winner = touch_data.get('winner', 'undetermined')
    fencer_intention = touch_data.get(f'{fencer_side}_intention', '')
    fencer_category = _determine_fencer_category(touch_data, fencer_side)

    did_win = (winner == fencer_side)

    # 1. First/Second Intention Effectiveness (in-box only)
    if fencer_category == 'in_box':
        if fencer_intention == 'first_intention':
            metrics['first_intention_attempts'] += 1
            if did_win:
                metrics['first_intention_wins'] += 1
        elif fencer_intention == 'second_intention':
            metrics['second_intention_attempts'] += 1
            if did_win:
                metrics['second_intention_wins'] += 1
    
    interval_analysis = fencer_data.get('interval_analysis', {})
    summary = interval_analysis.get('summary', {})

    if fencer_category == 'attack':
        # 2. Attack Promptness (quality metric based on velocity/acceleration)
        summary_metrics = fencer_data.get('summary_metrics', {}) or {}
        attacking_velocity = _get_metric_with_fallback(summary_metrics, 'attacking_velocity', 'avg_velocity', 'average_velocity')
        attacking_acceleration = _get_metric_with_fallback(summary_metrics, 'attacking_acceleration', 'avg_acceleration', 'average_acceleration')

        if attacking_velocity > 0 or attacking_acceleration > 0:
            velocity_score = normalize_value(attacking_velocity, 0, 4.0) * 100
            accel_score = normalize_value(attacking_acceleration, 0, 20.0) * 100
            promptness_score = (velocity_score * 0.4) + (accel_score * 0.6)
            metrics['attack_promptness_scores'].append(promptness_score)

        # 3. Attack Aggressiveness (volume of offensive actions)
        arm_extensions = len(fencer_data.get('extensions', []))
        launches = len(fencer_data.get('launches', []))
        offensive_actions = arm_extensions + launches
        aggressiveness_score = normalize_value(offensive_actions, 0, 5) * 100
        metrics['attack_aggressiveness_scores'].append(aggressiveness_score)

        # 4. Attack Distance Quality
        advance_analyses = interval_analysis.get('advance_analyses', [])
        for advance in advance_analyses:
            attack_info = advance.get('attack_info', {})
            if attack_info.get('has_attack', False):
                metrics['total_attacking_advances'] += 1
                if advance.get('good_attack_distance', False):
                    metrics['good_distance_advances'] += 1

        # 5. Attack Effectiveness
        metrics['attack_attempts'] += 1
        if did_win:
            metrics['attack_wins'] += 1

    # 6. Defense Distance Management (defense-category only, safe distance + spacing)
    if fencer_category == 'defense':
        retreat_analyses = interval_analysis.get('retreat_analyses', [])
        metrics['defense_total_retreat_intervals'] += len(retreat_analyses)
        for retreat in retreat_analyses:
            if retreat.get('maintained_safe_distance', False):
                metrics['defense_safe_distance_count'] += 1
            if retreat.get('consistent_spacing', False):
                metrics['defense_consistent_spacing_count'] += 1
    
        # 7. Counter Execution Rate (defense-category only)
        defense = summary.get('defense', {})
        metrics['total_counter_ops'] += defense.get('counter_opportunities', 0)
        metrics['counters_executed'] += defense.get('counters_executed', 0)
    
    # 8. Defense Effectiveness (defense-category only)
    if fencer_category == 'defense':
        metrics['defense_actions'] += 1
        if did_win:
            metrics['defense_wins'] += 1

def calculate_final_scores(metrics: Dict) -> Dict:
    """Calculate final 0-100 scores from accumulated metrics"""
    
    scores = {}
    
    # In-Box Metrics
    first_intention_attempts = metrics.get('first_intention_attempts', 0)
    first_intention_wins = metrics.get('first_intention_wins', 0)
    second_intention_attempts = metrics.get('second_intention_attempts', 0)
    second_intention_wins = metrics.get('second_intention_wins', 0)
    attack_promptness_scores = metrics.get('attack_promptness_scores', []) or []

    scores['first_intention_effectiveness'] = sanitize_value(
        (first_intention_wins / first_intention_attempts * 100)
        if first_intention_attempts > 0 else 0
    )
    
    scores['second_intention_effectiveness'] = sanitize_value(
        (second_intention_wins / second_intention_attempts * 100)
        if second_intention_attempts > 0 else 0
    )
    
    attack_promptness_value = None
    if attack_promptness_scores:
        attack_promptness_value = sum(attack_promptness_scores) / len(attack_promptness_scores)
    scores['attack_promptness'] = sanitize_value(attack_promptness_value) if attack_promptness_value is not None else 0.0
    
    # Attack Metrics  
    attack_aggressiveness_scores = metrics.get('attack_aggressiveness_scores', []) or []
    total_attacking_advances = metrics.get('total_attacking_advances', 0)
    good_distance_advances = metrics.get('good_distance_advances', 0)
    attack_attempts = metrics.get('attack_attempts', 0)
    attack_wins = metrics.get('attack_wins', 0)

    attack_aggressiveness_value = None
    if attack_aggressiveness_scores:
        attack_aggressiveness_value = sum(attack_aggressiveness_scores) / len(attack_aggressiveness_scores)
    scores['attack_aggressiveness'] = sanitize_value(attack_aggressiveness_value) if attack_aggressiveness_value is not None else 0.0
    
    attack_distance_value = (good_distance_advances / total_attacking_advances * 100) if total_attacking_advances > 0 else None
    scores['attack_distance_quality'] = sanitize_value(attack_distance_value) if attack_distance_value is not None else 0.0
    
    attack_effectiveness_value = (attack_wins / attack_attempts * 100) if attack_attempts > 0 else None
    scores['attack_effectiveness'] = sanitize_value(attack_effectiveness_value) if attack_effectiveness_value is not None else 0.0
    
    # Defense Metrics
    # Defense metrics
    defense_total_retreat_intervals = metrics.get('defense_total_retreat_intervals', 0)
    defense_safe_distance_count = metrics.get('defense_safe_distance_count', 0)
    defense_consistent_spacing_count = metrics.get('defense_consistent_spacing_count', 0)
    total_counter_ops = metrics.get('total_counter_ops', 0)
    counters_executed = metrics.get('counters_executed', 0)
    defense_actions = metrics.get('defense_actions', 0)
    defense_wins = metrics.get('defense_wins', 0)

    safe_rate = (defense_safe_distance_count / defense_total_retreat_intervals * 100) if defense_total_retreat_intervals > 0 else None
    spacing_rate = (defense_consistent_spacing_count / defense_total_retreat_intervals * 100) if defense_total_retreat_intervals > 0 else None
    defense_distance_value = (safe_rate + spacing_rate) / 2.0 if safe_rate is not None and spacing_rate is not None else None
    scores['defense_distance_management'] = sanitize_value(defense_distance_value) if defense_distance_value is not None else 0.0

    counter_execution_value = (counters_executed / total_counter_ops * 100) if total_counter_ops > 0 else None
    scores['counter_execution_rate'] = sanitize_value(counter_execution_value) if counter_execution_value is not None else 0.0
    
    defense_effectiveness_value = (defense_wins / defense_actions * 100) if defense_actions > 0 else None
    scores['defense_effectiveness'] = sanitize_value(defense_effectiveness_value) if defense_effectiveness_value is not None else 0.0
    
    # Calculate sub-scores and overall score
    in_box_components = []
    if first_intention_attempts > 0:
        in_box_components.append(scores['first_intention_effectiveness'])
    if second_intention_attempts > 0:
        in_box_components.append(scores['second_intention_effectiveness'])
    in_box_subscore = sanitize_value(sum(in_box_components) / len(in_box_components)) if in_box_components else None
    
    attack_components = []
    if attack_promptness_scores:
        attack_components.append(scores['attack_promptness'])
    if attack_aggressiveness_scores:
        attack_components.append(scores['attack_aggressiveness'])
    if total_attacking_advances > 0:
        attack_components.append(scores['attack_distance_quality'])
    if attack_attempts > 0:
        attack_components.append(scores['attack_effectiveness'])
    attack_subscore = sanitize_value(sum(attack_components) / len(attack_components)) if attack_components else None
    
    defense_components = []
    if defense_total_retreat_intervals > 0:
        defense_components.append(scores['defense_distance_management'])
    if total_counter_ops > 0:
        defense_components.append(scores['counter_execution_rate'])
    if defense_actions > 0:
        defense_components.append(scores['defense_effectiveness'])
    defense_subscore = sanitize_value(sum(defense_components) / len(defense_components)) if defense_components else None
    
    # Calculate overall score from available categories
    available_scores = []
    if in_box_subscore is not None:
        available_scores.append(in_box_subscore)
    if attack_subscore is not None:
        available_scores.append(attack_subscore)
    if defense_subscore is not None:
        available_scores.append(defense_subscore)
    
    scores['in_box_subscore'] = in_box_subscore
    scores['attack_subscore'] = attack_subscore
    scores['defense_subscore'] = defense_subscore
    scores['overall_score'] = sanitize_value(sum(available_scores) / len(available_scores)) if available_scores else 0.0
    
    return scores

def calculate_bout_type_statistics(match_data: List[Dict]) -> Dict:
    """Calculate bout type statistics similar to touch category view"""
    
    stats = {
        'left_fencer': {
            'attack': {'count': 0, 'wins': 0, 'win_rate': 0},
            'defense': {'count': 0, 'wins': 0, 'win_rate': 0},
            'in_box': {'count': 0, 'wins': 0, 'win_rate': 0}
        },
        'right_fencer': {
            'attack': {'count': 0, 'wins': 0, 'win_rate': 0},
            'defense': {'count': 0, 'wins': 0, 'win_rate': 0},
            'in_box': {'count': 0, 'wins': 0, 'win_rate': 0}
        }
    }
    
    for touch_data in match_data:
        winner = touch_data.get('winner', 'undetermined')
        left_category = _determine_fencer_category(touch_data, 'left')
        right_category = _determine_fencer_category(touch_data, 'right')
        
        # Count based on individual fencer categories
        if left_category in stats['left_fencer']:
            stats['left_fencer'][left_category]['count'] += 1
            if winner == 'left':
                stats['left_fencer'][left_category]['wins'] += 1
                
        if right_category in stats['right_fencer']:
            stats['right_fencer'][right_category]['count'] += 1
            if winner == 'right':
                stats['right_fencer'][right_category]['wins'] += 1
    
    # Calculate win rates
    for fencer in ['left_fencer', 'right_fencer']:
        for category in ['attack', 'defense', 'in_box']:
            count = stats[fencer][category]['count']
            wins = stats[fencer][category]['wins']
            stats[fencer][category]['win_rate'] = sanitize_value((wins / count * 100) if count > 0 else 0)
    
    return stats

def sanitize_data_structure(data):
    """Recursively sanitize a data structure to ensure JSON serializability"""
    if isinstance(data, dict):
        return {key: sanitize_data_structure(value) for key, value in data.items()}
    elif isinstance(data, list):
        return [sanitize_data_structure(item) for item in data]
    elif isinstance(data, bool):
        return data
    elif isinstance(data, (int, float)):
        return sanitize_value(data)
    elif data is None:
        return None
    elif isinstance(data, str):
        return data
    else:
        # Convert any other type to string as fallback
        return str(data)


VALID_FENCER_CATEGORIES = {'attack', 'defense', 'in_box'}


def _get_metric_with_fallback(summary_metrics: Dict, primary: str, *fallbacks: str) -> float:
    """Fetch a numeric metric from summary metrics using fallback keys."""
    if not summary_metrics:
        return 0.0
    for key in (primary, *fallbacks):
        value = summary_metrics.get(key)
        if isinstance(value, (int, float)) and not math.isnan(value) and not math.isinf(value):
            return float(value)
    return 0.0


def _determine_fencer_category(touch_data: Dict, fencer_side: str) -> str:
    """Resolve the tactical category for a fencer, inferring when explicit labels are missing."""
    explicit = touch_data.get(f'{fencer_side}_fencer_category')
    if explicit in VALID_FENCER_CATEGORIES:
        return explicit

    # Derive per-fencer categories deterministically from JSON movement data
    try:
        left_data = touch_data.get('left_data', {}) or {}
        right_data = touch_data.get('right_data', {}) or {}
        frame_range = touch_data.get('frame_range') or [0, 0]
        total_frames = None
        try:
            if isinstance(frame_range, (list, tuple)) and len(frame_range) >= 2:
                total_frames = int(frame_range[1]) - int(frame_range[0]) + 1
        except Exception:
            total_frames = None
        fps = touch_data.get('fps') or 30
        left_cat, right_cat = classify_bout_categories(left_data, right_data, total_frames, fps)
        return left_cat if fencer_side == 'left' else right_cat
    except Exception:
        # Final fallback: use bout_type if valid, else default to in_box
        bout_type = touch_data.get('bout_type')
        if bout_type in VALID_FENCER_CATEGORIES:
            return bout_type
        return 'in_box'

def calculate_inbox_analysis(match_data: List[Dict]) -> Dict:
    """Calculate opening-moment metrics across in-box bouts for each fencer (per-fencer category)."""
    left_metrics = {
        'reaction_times': [],
        'initial_accelerations': [],
        'total_extensions': 0,
        'total_lunges': 0,
        'first_intention_count': 0,
        'second_intention_count': 0,
        'bouts_count': 0,
        'initial_velocities': []
    }
    right_metrics = {
        'reaction_times': [],
        'initial_accelerations': [],
        'total_extensions': 0,
        'total_lunges': 0,
        'first_intention_count': 0,
        'second_intention_count': 0,
        'bouts_count': 0,
        'initial_velocities': []
    }

    for touch_data in match_data:
        fps = touch_data.get('fps', 30)

        if _determine_fencer_category(touch_data, 'left') == 'in_box':
            left_data = touch_data.get('left_data', {}) or {}
            left_metrics['bouts_count'] += 1
            first_step = left_data.get('first_step', {}) or {}
            init_time = first_step.get('init_time')
            if init_time is not None:
                left_metrics['reaction_times'].append(init_time)
            initial_velocity = first_step.get('velocity')
            if initial_velocity is not None:
                left_metrics['initial_velocities'].append(initial_velocity)
            initial_acceleration = first_step.get('acceleration')
            if initial_acceleration is not None:
                left_metrics['initial_accelerations'].append(initial_acceleration)
            # Total extensions across the whole bout (prefer detailed list, fallback to sec intervals)
            try:
                total_ext = len(left_data.get('extensions', []) or [])
                if total_ext == 0:
                    total_ext = len(left_data.get('arm_extensions_sec', []) or [])
                left_metrics['total_extensions'] += int(total_ext)
            except Exception:
                pass
            # Total lunges across the whole bout (prefer launches list, fallback to has_launch)
            try:
                launches_list = left_data.get('launches', []) or []
                if isinstance(launches_list, list) and len(launches_list) > 0:
                    left_metrics['total_lunges'] += int(len(launches_list))
                else:
                    if bool(left_data.get('has_launch')):
                        left_metrics['total_lunges'] += 1
            except Exception:
                pass
            left_intention = touch_data.get('left_intention', '')
            if left_intention == 'first_intention':
                left_metrics['first_intention_count'] += 1
            elif left_intention == 'second_intention':
                left_metrics['second_intention_count'] += 1

        if _determine_fencer_category(touch_data, 'right') == 'in_box':
            right_data = touch_data.get('right_data', {}) or {}
            right_metrics['bouts_count'] += 1
            first_step = right_data.get('first_step', {}) or {}
            init_time = first_step.get('init_time')
            if init_time is not None:
                right_metrics['reaction_times'].append(init_time)
            initial_velocity = first_step.get('velocity')
            if initial_velocity is not None:
                right_metrics['initial_velocities'].append(initial_velocity)
            initial_acceleration = first_step.get('acceleration')
            if initial_acceleration is not None:
                right_metrics['initial_accelerations'].append(initial_acceleration)
            # Total extensions across the whole bout (prefer detailed list, fallback to sec intervals)
            try:
                total_ext = len(right_data.get('extensions', []) or [])
                if total_ext == 0:
                    total_ext = len(right_data.get('arm_extensions_sec', []) or [])
                right_metrics['total_extensions'] += int(total_ext)
            except Exception:
                pass
            # Total lunges across the whole bout (prefer launches list, fallback to has_launch)
            try:
                launches_list = right_data.get('launches', []) or []
                if isinstance(launches_list, list) and len(launches_list) > 0:
                    right_metrics['total_lunges'] += int(len(launches_list))
                else:
                    if bool(right_data.get('has_launch')):
                        right_metrics['total_lunges'] += 1
            except Exception:
                pass
            right_intention = touch_data.get('right_intention', '')
            if right_intention == 'first_intention':
                right_metrics['first_intention_count'] += 1
            elif right_intention == 'second_intention':
                right_metrics['second_intention_count'] += 1

    return {
        'left_fencer': {
            'reaction_time': sanitize_value(sum(left_metrics['reaction_times']) / len(left_metrics['reaction_times'])) if left_metrics['reaction_times'] else 0,
            'initial_velocity': sanitize_value(sum(left_metrics['initial_velocities']) / len(left_metrics['initial_velocities'])) if left_metrics['initial_velocities'] else 0,
            'initial_acceleration': sanitize_value(sum(left_metrics['initial_accelerations']) / len(left_metrics['initial_accelerations'])) if left_metrics['initial_accelerations'] else 0,
            'total_extensions': left_metrics['total_extensions'],
            'total_lunges': left_metrics['total_lunges'],
            'first_intention_starts': left_metrics['first_intention_count'],
            'second_intention_starts': left_metrics['second_intention_count']
        },
        'right_fencer': {
            'reaction_time': sanitize_value(sum(right_metrics['reaction_times']) / len(right_metrics['reaction_times'])) if right_metrics['reaction_times'] else 0,
            'initial_velocity': sanitize_value(sum(right_metrics['initial_velocities']) / len(right_metrics['initial_velocities'])) if right_metrics['initial_velocities'] else 0,
            'initial_acceleration': sanitize_value(sum(right_metrics['initial_accelerations']) / len(right_metrics['initial_accelerations'])) if right_metrics['initial_accelerations'] else 0,
            'total_extensions': right_metrics['total_extensions'],
            'total_lunges': right_metrics['total_lunges'],
            'first_intention_starts': right_metrics['first_intention_count'],
            'second_intention_starts': right_metrics['second_intention_count']
        },
        'meta': {
            'left_count': left_metrics['bouts_count'],
            'right_count': right_metrics['bouts_count']
        }
    }

def calculate_attack_analysis(match_data: List[Dict]) -> Dict:
    """Calculate detailed Attack analysis metrics for mirror bar chart (per-fencer attack bouts only)."""
    left_metrics = {
        'attack_velocities': [],
        'total_attacking_advances': 0,
        'good_distance_advances': 0,
        'simple_attacks': 0,
        'compound_attacks': 0,
        'holding_attacks': 0,
        'preparations': 0,
        'steady_tempo': 0,
        'variable_tempo': 0,
        'broken_tempo': 0,
        'bouts_count': 0
    }
    right_metrics = {
        'attack_velocities': [],
        'total_attacking_advances': 0,
        'good_distance_advances': 0,
        'simple_attacks': 0,
        'compound_attacks': 0,
        'holding_attacks': 0,
        'preparations': 0,
        'steady_tempo': 0,
        'variable_tempo': 0,
        'broken_tempo': 0,
        'bouts_count': 0
    }

    for touch_data in match_data:
        if _determine_fencer_category(touch_data, 'left') == 'attack':
            left_data = touch_data.get('left_data', {}) or {}
            left_metrics['bouts_count'] += 1
            summary_metrics = left_data.get('summary_metrics', {}) or {}
            # Use attack-specific velocity only; do not fall back to overall/avg velocity
            velocity = summary_metrics.get('attacking_velocity')
            if not isinstance(velocity, (int, float)) or math.isnan(velocity) or math.isinf(velocity):
                velocity = 0.0
            if velocity > 0:
                left_metrics['attack_velocities'].append(velocity)
            advance_analyses = left_data.get('interval_analysis', {}).get('advance_analyses', []) or []
            for advance in advance_analyses:
                attack_info = advance.get('attack_info', {}) or {}
                if attack_info.get('has_attack'):
                    left_metrics['total_attacking_advances'] += 1
                    if advance.get('good_attack_distance', False):
                        left_metrics['good_distance_advances'] += 1
                    # Count tempo only for advances where an attack occurred
                    tempo_type = advance.get('tempo_type')
                    if tempo_type == 'steady_tempo':
                        left_metrics['steady_tempo'] += 1
                    elif tempo_type == 'variable_tempo':
                        left_metrics['variable_tempo'] += 1
                    elif tempo_type == 'broken_tempo':
                        left_metrics['broken_tempo'] += 1
            summary = left_data.get('interval_analysis', {}).get('summary', {}) or {}
            attacks = (summary.get('attacks', {}) or {})
            left_metrics['simple_attacks'] += int(attacks.get('simple', 0) or 0)
            left_metrics['compound_attacks'] += int(attacks.get('compound', 0) or 0)
            left_metrics['holding_attacks'] += int(attacks.get('holding', 0) or 0)
            left_metrics['preparations'] += int(attacks.get('preparations', 0) or 0)

        if _determine_fencer_category(touch_data, 'right') == 'attack':
            right_data = touch_data.get('right_data', {}) or {}
            right_metrics['bouts_count'] += 1
            summary_metrics = right_data.get('summary_metrics', {}) or {}
            # Use attack-specific velocity only; do not fall back to overall/avg velocity
            velocity = summary_metrics.get('attacking_velocity')
            if not isinstance(velocity, (int, float)) or math.isnan(velocity) or math.isinf(velocity):
                velocity = 0.0
            if velocity > 0:
                right_metrics['attack_velocities'].append(velocity)
            advance_analyses = right_data.get('interval_analysis', {}).get('advance_analyses', []) or []
            for advance in advance_analyses:
                attack_info = advance.get('attack_info', {}) or {}
                if attack_info.get('has_attack'):
                    right_metrics['total_attacking_advances'] += 1
                    if advance.get('good_attack_distance', False):
                        right_metrics['good_distance_advances'] += 1
                    # Count tempo only for advances where an attack occurred
                    tempo_type = advance.get('tempo_type')
                    if tempo_type == 'steady_tempo':
                        right_metrics['steady_tempo'] += 1
                    elif tempo_type == 'variable_tempo':
                        right_metrics['variable_tempo'] += 1
                    elif tempo_type == 'broken_tempo':
                        right_metrics['broken_tempo'] += 1
            summary = right_data.get('interval_analysis', {}).get('summary', {}) or {}
            attacks = (summary.get('attacks', {}) or {})
            right_metrics['simple_attacks'] += int(attacks.get('simple', 0) or 0)
            right_metrics['compound_attacks'] += int(attacks.get('compound', 0) or 0)
            right_metrics['holding_attacks'] += int(attacks.get('holding', 0) or 0)
            right_metrics['preparations'] += int(attacks.get('preparations', 0) or 0)

    return {
        'left_fencer': {
            'avg_attack_velocity': sanitize_value(sum(left_metrics['attack_velocities']) / len(left_metrics['attack_velocities'])) if left_metrics['attack_velocities'] else 0,
            'attack_distance_quality': sanitize_value((left_metrics['good_distance_advances'] / left_metrics['total_attacking_advances'] * 100) if left_metrics['total_attacking_advances'] > 0 else 0),
            'simple_attacks': left_metrics['simple_attacks'],
            'compound_attacks': left_metrics['compound_attacks'],
            'holding_attacks': left_metrics['holding_attacks'],
            'preparations': left_metrics['preparations'],
            'steady_tempo_attacks': left_metrics['steady_tempo'],
            'variable_tempo_attacks': left_metrics['variable_tempo'],
            'broken_tempo_attacks': left_metrics['broken_tempo']
        },
        'right_fencer': {
            'avg_attack_velocity': sanitize_value(sum(right_metrics['attack_velocities']) / len(right_metrics['attack_velocities'])) if right_metrics['attack_velocities'] else 0,
            'attack_distance_quality': sanitize_value((right_metrics['good_distance_advances'] / right_metrics['total_attacking_advances'] * 100) if right_metrics['total_attacking_advances'] > 0 else 0),
            'simple_attacks': right_metrics['simple_attacks'],
            'compound_attacks': right_metrics['compound_attacks'],
            'holding_attacks': right_metrics['holding_attacks'],
            'preparations': right_metrics['preparations'],
            'steady_tempo_attacks': right_metrics['steady_tempo'],
            'variable_tempo_attacks': right_metrics['variable_tempo'],
            'broken_tempo_attacks': right_metrics['broken_tempo']
        },
        'meta': {
            'left_count': left_metrics['bouts_count'],
            'right_count': right_metrics['bouts_count']
        }
    }

def calculate_defense_analysis(match_data: List[Dict]) -> Dict:
    """Calculate detailed Defense analysis metrics for mirror bar chart (per-fencer defense bouts only)."""
    left_metrics = {
        'total_retreats': 0,
        'total_retreat_intervals': 0,
        'safe_distance_count': 0,
        'consistent_spacing_count': 0,
        'counter_opportunities': 0,
        'counters_executed': 0,
        'bouts_count': 0
    }
    right_metrics = {
        'total_retreats': 0,
        'total_retreat_intervals': 0,
        'safe_distance_count': 0,
        'consistent_spacing_count': 0,
        'counter_opportunities': 0,
        'counters_executed': 0,
        'bouts_count': 0
    }

    for touch_data in match_data:
        if _determine_fencer_category(touch_data, 'left') == 'defense':
            left_data = touch_data.get('left_data', {}) or {}
            left_metrics['bouts_count'] += 1
            movement_data = left_data.get('movement_data', {}) or {}
            retreat_intervals = movement_data.get('retreat_intervals', []) or []
            retreat_analyses = left_data.get('interval_analysis', {}).get('retreat_analyses', []) or []
            left_metrics['total_retreats'] += len(retreat_intervals)
            left_metrics['total_retreat_intervals'] += len(retreat_analyses)
            for retreat in retreat_analyses:
                if retreat.get('maintained_safe_distance', False):
                    left_metrics['safe_distance_count'] += 1
                if retreat.get('consistent_spacing', False):
                    left_metrics['consistent_spacing_count'] += 1
            defense = left_data.get('interval_analysis', {}).get('summary', {}).get('defense', {}) or {}
            left_metrics['counter_opportunities'] += int(defense.get('counter_opportunities', 0) or 0)
            left_metrics['counters_executed'] += int(defense.get('counters_executed', 0) or 0)

        if _determine_fencer_category(touch_data, 'right') == 'defense':
            right_data = touch_data.get('right_data', {}) or {}
            right_metrics['bouts_count'] += 1
            movement_data = right_data.get('movement_data', {}) or {}
            retreat_intervals = movement_data.get('retreat_intervals', []) or []
            retreat_analyses = right_data.get('interval_analysis', {}).get('retreat_analyses', []) or []
            right_metrics['total_retreats'] += len(retreat_intervals)
            right_metrics['total_retreat_intervals'] += len(retreat_analyses)
            for retreat in retreat_analyses:
                if retreat.get('maintained_safe_distance', False):
                    right_metrics['safe_distance_count'] += 1
                if retreat.get('consistent_spacing', False):
                    right_metrics['consistent_spacing_count'] += 1
            defense = right_data.get('interval_analysis', {}).get('summary', {}).get('defense', {}) or {}
            right_metrics['counter_opportunities'] += int(defense.get('counter_opportunities', 0) or 0)
            right_metrics['counters_executed'] += int(defense.get('counters_executed', 0) or 0)

    return {
        'left_fencer': {
            'total_retreats': left_metrics['total_retreats'],
            'safe_distance_rate': sanitize_value((left_metrics['safe_distance_count'] / left_metrics['total_retreat_intervals'] * 100) if left_metrics['total_retreat_intervals'] > 0 else 0),
            'consistent_spacing_rate': sanitize_value((left_metrics['consistent_spacing_count'] / left_metrics['total_retreat_intervals'] * 100) if left_metrics['total_retreat_intervals'] > 0 else 0),
            'counter_opportunities_created': left_metrics['counter_opportunities'],
            'counter_execution_rate': sanitize_value((left_metrics['counters_executed'] / left_metrics['counter_opportunities'] * 100) if left_metrics['counter_opportunities'] > 0 else 0)
        },
        'right_fencer': {
            'total_retreats': right_metrics['total_retreats'],
            'safe_distance_rate': sanitize_value((right_metrics['safe_distance_count'] / right_metrics['total_retreat_intervals'] * 100) if right_metrics['total_retreat_intervals'] > 0 else 0),
            'consistent_spacing_rate': sanitize_value((right_metrics['consistent_spacing_count'] / right_metrics['total_retreat_intervals'] * 100) if right_metrics['total_retreat_intervals'] > 0 else 0),
            'counter_opportunities_created': right_metrics['counter_opportunities'],
            'counter_execution_rate': sanitize_value((right_metrics['counters_executed'] / right_metrics['counter_opportunities'] * 100) if right_metrics['counter_opportunities'] > 0 else 0)
        },
        'meta': {
            'left_count': left_metrics['bouts_count'],
            'right_count': right_metrics['bouts_count']
        }
    }

def _normalize_with_baseline(value: float, min_baseline: float, max_baseline: float, is_lower_better: bool = False) -> float:
    """
    Normalize a value to 0-10 scale using proper baseline ranges.
    
    Args:
        value: Raw metric value
        min_baseline: Minimum expected value (0 for most metrics)
        max_baseline: Maximum expected value (varies by metric type)
        is_lower_better: True for metrics like reaction time where lower is better
    
    Returns:
        float: Normalized value between 0-10 (0 for zero values, 1-10 for others)
    """
    value = sanitize_value(value)
    
    # Special case: if value is 0 (no count/percentage), return 0
    if value == 0:
        return 0.0
    
    # Clamp value to baseline range
    clamped_value = max(min_baseline, min(max_baseline, value))
    
    if is_lower_better:
        # For reaction time: lower values get higher scores
        # Invert the scale: max_baseline -> 1, min_baseline -> 10
        if max_baseline == min_baseline:
            return 5.5  # Neutral if no range
        normalized = 1 + (9 * (max_baseline - clamped_value) / (max_baseline - min_baseline))
    else:
        # For most metrics: higher values get higher scores  
        # Normal scale: min_baseline -> 1, max_baseline -> 10
        if max_baseline == min_baseline:
            return 5.5  # Neutral if no range
        normalized = 1 + (9 * (clamped_value - min_baseline) / (max_baseline - min_baseline))
    
    return max(1.0, min(10.0, normalized))

def _calculate_relative_advantage_bars(left_value: float, right_value: float, metric_type: str, baseline_range: tuple) -> tuple:
    """
    Calculate relative advantage bar lengths using proper baseline normalization.
    
    Args:
        left_value: Left fencer's raw metric value
        right_value: Right fencer's raw metric value  
        metric_type: Type of metric comparison ('higher_better' or 'lower_better')
        baseline_range: (min_baseline, max_baseline) for this metric type
    
    Returns:
        tuple: (left_bar_length, right_bar_length) both values 1-10
    """
    
    min_baseline, max_baseline = baseline_range
    is_lower_better = (metric_type == 'lower_better')
    
    # Normalize both values using the baseline
    left_normalized = _normalize_with_baseline(left_value, min_baseline, max_baseline, is_lower_better)
    right_normalized = _normalize_with_baseline(right_value, min_baseline, max_baseline, is_lower_better)
    
    return (left_normalized, right_normalized)

def _prepare_category_chart_data(left_data: Dict, right_data: Dict, category: str) -> Dict:
    """
    Prepare chart data for mirror bar charts showing relative advantage.
    
    Returns:
        Dict with structure:
        {
            'chart_data': {
                'left_fencer': {metric: bar_length, ...},
                'right_fencer': {metric: bar_length, ...}
            },
            'display_data': {
                'left_fencer': {metric: 'raw_value unit', ...},
                'right_fencer': {metric: 'raw_value unit', ...}
            }
        }
    """
    
    chart_left = {}
    chart_right = {}
    display_left = {}
    display_right = {}
    
    if category == 'in_box':
        # In-box category metrics with proper baseline ranges
        metrics_config = [
            ('reaction_time', 'Avg Reaction Time', 'lower_better', 's', (0, 2)),        # 0-2 seconds
            ('initial_velocity', 'Initial Velocity', 'higher_better', 'm/s', (0, 10)),      # 0-10 m/s
            ('initial_acceleration', 'Initial Acceleration', 'higher_better', 'm/s²', (0, 20)), # 0-20 m/s²
            ('total_extensions', 'Total Extensions', 'higher_better', 'count', (0, 50)),        # total count
            ('total_lunges', 'Total Lunges', 'higher_better', 'count', (0, 50)),           # total count
            ('first_intention_starts', 'First Intention', 'higher_better', 'count', (0, 20)),  # 0-20 count
            ('second_intention_starts', 'Second Intention', 'higher_better', 'count', (0, 20))  # 0-20 count
        ]
        
        for metric_key, label, metric_type, unit, baseline_range in metrics_config:
            left_val = left_data.get(metric_key, 0)
            right_val = right_data.get(metric_key, 0)

            # Calculate relative advantage bars with proper baseline
            left_bar, right_bar = _calculate_relative_advantage_bars(left_val, right_val, metric_type, baseline_range)
            chart_left[label] = left_bar
            chart_right[label] = right_bar

            # Format display values with units
            if metric_key == 'reaction_time':
                display_left[label] = f"{left_val:.3f}{unit}"
                display_right[label] = f"{right_val:.3f}{unit}"
            elif metric_key in ['initial_velocity', 'initial_acceleration']:
                display_left[label] = f"{left_val:.2f}{unit}"
                display_right[label] = f"{right_val:.2f}{unit}"
            else:
                display_left[label] = f"{int(left_val)}{unit}"
                display_right[label] = f"{int(right_val)}{unit}"
        
    elif category == 'attack':
        # Attack category metrics with proper baseline ranges
        metrics_config = [
            ('avg_attack_velocity', 'Attack Velocity', 'higher_better', 'm/s', (0, 10)),     # 0-10 m/s
            ('attack_distance_quality', 'Distance Quality', 'higher_better', '%', (0, 100)),  # 0-100%
            ('simple_attacks', 'Simple Attacks', 'higher_better', 'count', (0, 20)),           # 0-20 count
            ('compound_attacks', 'Compound Attacks', 'higher_better', 'count', (0, 20)),         # 0-20 count
            ('holding_attacks', 'Holding Attacks', 'higher_better', 'count', (0, 20)),          # 0-20 count
            ('preparations', 'Preparations', 'higher_better', 'count', (0, 20)),            # 0-20 count
            ('broken_tempo_attacks', 'Broken Tempo', 'higher_better', 'count', (0, 20)),       # 0-20 count
            ('variable_tempo_attacks', 'Variable Tempo', 'higher_better', 'count', (0, 20)),     # 0-20 count
            ('steady_tempo_attacks', 'Steady Tempo', 'higher_better', 'count', (0, 20))        # 0-20 count
        ]
        
        for metric_key, label, metric_type, unit, baseline_range in metrics_config:
            left_val = left_data.get(metric_key, 0)
            right_val = right_data.get(metric_key, 0)

            # Calculate relative advantage bars with proper baseline
            left_bar, right_bar = _calculate_relative_advantage_bars(left_val, right_val, metric_type, baseline_range)
            chart_left[label] = left_bar
            chart_right[label] = right_bar

            # Format display values with units
            if metric_key == 'avg_attack_velocity':
                display_left[label] = f"{left_val:.2f}{unit}"
                display_right[label] = f"{right_val:.2f}{unit}"
            elif metric_key == 'attack_distance_quality':
                display_left[label] = f"{left_val:.1f}{unit}"
                display_right[label] = f"{right_val:.1f}{unit}"
            else:
                display_left[label] = f"{int(left_val)}{unit}"
                display_right[label] = f"{int(right_val)}{unit}"
                
    elif category == 'defense':
        # Defense category metrics with proper baseline ranges
        metrics_config = [
            ('total_retreats', 'Total Retreats', 'higher_better', 'count', (0, 20)),           # 0-20 count
            ('safe_distance_rate', 'Safe Distance Rate', 'higher_better', '%', (0, 100)),     # 0-100%
            ('consistent_spacing_rate', 'Spacing Consistency', 'higher_better', '%', (0, 100)), # 0-100%
            ('counter_opportunities_created', 'Counter Opportunities', 'higher_better', 'count', (0, 20)), # 0-20 count
            ('counter_execution_rate', 'Counter Execution Rate', 'higher_better', '%', (0, 100))  # 0-100%
        ]
        
        for metric_key, label, metric_type, unit, baseline_range in metrics_config:
            left_val = left_data.get(metric_key, 0)
            right_val = right_data.get(metric_key, 0)

            # Calculate relative advantage bars with proper baseline
            left_bar, right_bar = _calculate_relative_advantage_bars(left_val, right_val, metric_type, baseline_range)
            chart_left[label] = left_bar
            chart_right[label] = right_bar

            # Format display values with units
            if unit == '%':
                display_left[label] = f"{left_val:.1f}{unit}"
                display_right[label] = f"{right_val:.1f}{unit}"
            else:
                display_left[label] = f"{int(left_val)}{unit}"
                display_right[label] = f"{int(right_val)}{unit}"
    
    return {
        'chart_data': {
            'left_fencer': chart_left,
            'right_fencer': chart_right
        },
        'display_data': {
            'left_fencer': display_left,
            'right_fencer': display_right
        }
    }

def build_detailed_analysis(inbox_analysis: Dict, attack_analysis: Dict, defense_analysis: Dict) -> Dict:
    """
    Build mirror bar chart data with relative advantage visualization and original data display.
    
    Returns structured data for both chart visualization and raw data display.
    """
    
    # Process in-box category
    left_inbox_data = inbox_analysis.get('left_fencer', {})
    right_inbox_data = inbox_analysis.get('right_fencer', {})
    
    if left_inbox_data or right_inbox_data:
        inbox_chart_data = _prepare_category_chart_data(left_inbox_data, right_inbox_data, 'in_box')
        inbox_result = {
            'left_fencer': inbox_chart_data['chart_data']['left_fencer'],
            'right_fencer': inbox_chart_data['chart_data']['right_fencer'],
            'display_data': inbox_chart_data['display_data'],
            'meta': inbox_analysis.get('meta', {})
        }
    else:
        inbox_result = {
            'left_fencer': None,
            'right_fencer': None,
            'display_data': {'left_fencer': {}, 'right_fencer': {}},
            'meta': inbox_analysis.get('meta', {})
        }
    
    # Process attack category  
    left_attack_data = attack_analysis.get('left_fencer', {})
    right_attack_data = attack_analysis.get('right_fencer', {})
    
    if left_attack_data or right_attack_data:
        attack_chart_data = _prepare_category_chart_data(left_attack_data, right_attack_data, 'attack')
        attack_result = {
            'left_fencer': attack_chart_data['chart_data']['left_fencer'],
            'right_fencer': attack_chart_data['chart_data']['right_fencer'],
            'display_data': attack_chart_data['display_data'],
            'meta': attack_analysis.get('meta', {})
        }
    else:
        attack_result = {
            'left_fencer': None,
            'right_fencer': None,
            'display_data': {'left_fencer': {}, 'right_fencer': {}},
            'meta': attack_analysis.get('meta', {})
        }
    
    # Process defense category
    left_defense_data = defense_analysis.get('left_fencer', {})
    right_defense_data = defense_analysis.get('right_fencer', {})
    
    if left_defense_data or right_defense_data:
        defense_chart_data = _prepare_category_chart_data(left_defense_data, right_defense_data, 'defense')
        defense_result = {
            'left_fencer': defense_chart_data['chart_data']['left_fencer'],
            'right_fencer': defense_chart_data['chart_data']['right_fencer'],
            'display_data': defense_chart_data['display_data'],
            'meta': defense_analysis.get('meta', {})
        }
    else:
        defense_result = {
            'left_fencer': None,
            'right_fencer': None,
            'display_data': {'left_fencer': {}, 'right_fencer': {}},
            'meta': defense_analysis.get('meta', {})
        }
    
    return {
        'in_box': inbox_result,
        'attack': attack_result,
        'defense': defense_result
}


def _build_mirror_chart_image(category_key: str, category_data: Dict) -> Optional[str]:
    """Render a mirror bar chart for a category and return it as a base64 image string."""
    if not category_data:
        return None

    left_series = category_data.get('left_fencer') or {}
    right_series = category_data.get('right_fencer') or {}
    display_data = category_data.get('display_data') or {}
    left_display = display_data.get('left_fencer', {})
    right_display = display_data.get('right_fencer', {})

    if not left_series or not right_series:
        return None

    labels = list(left_series.keys())
    if not labels:
        return None

    left_values: List[float] = []
    right_values: List[float] = []
    left_colors: List[str] = []
    right_colors: List[str] = []
    left_labels: List[str] = []
    right_labels: List[str] = []

    advantage_color = '#FF4444'
    baseline_color = '#B0B0B0'

    for label in labels:
        left_raw = float(left_series.get(label, 0) or 0)
        right_raw = float(right_series.get(label, 0) or 0)

        left_values.append(-abs(left_raw))
        right_values.append(right_raw)

        if left_raw > right_raw:
            left_colors.append(advantage_color)
            right_colors.append(baseline_color)
        elif right_raw > left_raw:
            left_colors.append(baseline_color)
            right_colors.append(advantage_color)
        else:
            left_colors.append(baseline_color)
            right_colors.append(baseline_color)

        left_labels.append(str(left_display.get(label, '') or ''))
        right_labels.append(str(right_display.get(label, '') or ''))

    has_data = any(abs(value) > 0.01 for value in left_values + right_values)
    if not has_data:
        return None

    bar_limit = 10
    fig_height = max(2.5, 0.6 * len(labels))
    fig, ax = plt.subplots(figsize=(8, fig_height))

    y_positions = range(len(labels))
    ax.barh(y_positions, left_values, color=left_colors, edgecolor='white')
    ax.barh(y_positions, right_values, color=right_colors, edgecolor='white')

    ax.set_yticks(list(y_positions))
    ax.set_yticklabels(labels)
    ax.set_xticks([])
    ax.set_xlim(-bar_limit, bar_limit)
    ax.axvline(0, color='#333333', linewidth=1)
    ax.grid(axis='x', color=(0, 0, 0, 0.1), linestyle='--', linewidth=0.5)
    legend_handles = [
        Patch(facecolor=advantage_color, edgecolor='white', label='Advantage'),
        Patch(facecolor=baseline_color, edgecolor='white', label='Baseline')
    ]
    ax.legend(handles=legend_handles, loc='lower right')
    ax.set_title('← Left Fencer | Right Fencer →', fontsize=12, pad=12)

    def _compute_position(value: float, side: str) -> float:
        minimum_offset = 0.6
        if side == 'left':
            if value < -0.5:
                return value / 2.0
            return -minimum_offset
        if value > 0.5:
            return value / 2.0
        return minimum_offset

    for y, value, raw_text, color in zip(y_positions, left_values, left_labels, left_colors):
        raw_text = raw_text.strip()
        if not raw_text:
            continue
        text_color = '#FFFFFF' if color == advantage_color and abs(value) > 1.2 else '#1f2933'
        text_x = _compute_position(value, 'left')
        ax.text(text_x, y, raw_text, ha='center', va='center', fontsize=8, color=text_color)

    for y, value, raw_text, color in zip(y_positions, right_values, right_labels, right_colors):
        raw_text = raw_text.strip()
        if not raw_text:
            continue
        text_color = '#FFFFFF' if color == advantage_color and abs(value) > 1.2 else '#1f2933'
        text_x = _compute_position(value, 'right')
        ax.text(text_x, y, raw_text, ha='center', va='center', fontsize=8, color=text_color)

    plt.tight_layout()
    buffer = io.BytesIO()
    fig.savefig(buffer, format='png', dpi=160)
    plt.close(fig)

    encoded = base64.b64encode(buffer.getvalue()).decode('ascii')
    return f'data:image/png;base64,{encoded}'


def _build_touch_summary(match_data: List[Dict]) -> List[Dict]:
    summary: List[Dict[str, Any]] = []
    for idx, touch in enumerate(match_data):
        left_category_key = _determine_fencer_category(touch, 'left')
        right_category_key = _determine_fencer_category(touch, 'right')
        left_category = CATEGORY_LABELS.get(left_category_key, left_category_key.title())
        right_category = CATEGORY_LABELS.get(right_category_key, right_category_key.title())

        winner = touch.get('winner', 'undetermined')
        if winner == 'left':
            result_label = 'Left Fencer'
        elif winner == 'right':
            result_label = 'Right Fencer'
        elif winner in {'double', 'double_touch'}:
            result_label = 'Double Touch'
        elif winner in {'simultaneous', 'no_result'}:
            result_label = 'Simultaneous'
        else:
            result_label = 'No Score'

        summary.append({
            'touch_number': idx + 1,
            'left_category': left_category,
            'right_category': right_category,
            'result_label': result_label
        })

    return summary


def generate_mirror_chart_images(detailed_analysis: Dict) -> Dict[str, Optional[str]]:
    """Generate mirror bar chart images for each category."""
    images = {}
    for category in ['in_box', 'attack', 'defense']:
        category_data = detailed_analysis.get(category) or {}
        images[category] = _build_mirror_chart_image(category, category_data)
    return images

_ELLIPSIS_LINE_RE = re.compile(r'(?m)^\s*\.\.\.\s*$')
_TRAILING_COMMA_RE = re.compile(r',(?=\s*[}\]])')
_ZERO_WIDTH_RE = re.compile('[\u200b\ufeff]')


def sanitize_json_text(text: str) -> str:
    """Best-effort cleanup for slightly malformed JSON snippets."""
    cleaned = text.strip()
    if not cleaned:
        return cleaned

    cleaned = _ZERO_WIDTH_RE.sub('', cleaned)
    cleaned = _ELLIPSIS_LINE_RE.sub('', cleaned)
    cleaned = _TRAILING_COMMA_RE.sub('', cleaned)

    def _escape_unescaped_newlines(s: str) -> str:
        result_chars: List[str] = []
        in_string = False
        escape = False
        for ch in s:
            if ch == '"' and not escape:
                in_string = not in_string
                result_chars.append(ch)
                escape = False
                continue
            if ch == '\\' and not escape:
                result_chars.append(ch)
                escape = True
                continue
            if ch == '\n' and in_string:
                result_chars.append('\\n')
                escape = False
                continue
            if ch == '\r' and in_string:
                result_chars.append('\\r')
                escape = False
                continue
            if escape:
                escape = False
            result_chars.append(ch)
        return ''.join(result_chars)

    cleaned = _escape_unescaped_newlines(cleaned)

    # Remove trailing characters after the final matching brace if present
    last_brace = cleaned.rfind('}')
    last_bracket = cleaned.rfind(']')
    cutoff = max(last_brace, last_bracket)
    if cutoff != -1 and cutoff + 1 < len(cleaned):
        cleaned = cleaned[:cutoff + 1]

    return cleaned.strip()


def extract_json_from_markdown(content: str) -> str:
    """Extract JSON content from markdown-formatted text with robust handling"""
    if not content:
        return ""
    
    content = content.strip()
    
    # Method 1: Handle ```json ... ``` format
    if '```json' in content:
        try:
            start_marker = '```json'
            end_marker = '```'
            
            start_pos = content.find(start_marker)
            if start_pos != -1:
                # Find the start of JSON content (after ```json and possible newlines)
                json_start = start_pos + len(start_marker)
                # Skip any whitespace/newlines after ```json
                while json_start < len(content) and content[json_start] in ' \n\r\t':
                    json_start += 1
                
                # Find the ending ```
                end_pos = content.find(end_marker, json_start)
                if end_pos != -1:
                    json_content = content[json_start:end_pos].strip()
                    if json_content:
                        return json_content
        except Exception:
            pass
    
    # Method 2: Handle ``` ... ``` format (generic code block)
    if '```' in content:
        try:
            # Find first ```
            first_triple = content.find('```')
            if first_triple != -1:
                # Find content after first ```
                json_start = first_triple + 3
                # Skip any language identifier on the same line
                newline_pos = content.find('\n', json_start)
                if newline_pos != -1 and newline_pos - json_start < 20:  # Language identifier should be short
                    json_start = newline_pos + 1
                
                # Find the closing ```
                second_triple = content.find('```', json_start)
                if second_triple != -1:
                    json_content = content[json_start:second_triple].strip()
                    if json_content and (json_content.startswith('{') or json_content.startswith('[')):
                        return json_content
        except Exception:
            pass
    
    # Method 3: Find JSON object boundaries in the raw content
    try:
        if '{' in content:
            start_idx = content.find('{')
            
            # Find the matching closing brace using bracket counting
            brace_count = 0
            end_idx = -1
            
            for i in range(start_idx, len(content)):
                char = content[i]
                if char == '{':
                    brace_count += 1
                elif char == '}':
                    brace_count -= 1
                    if brace_count == 0:
                        end_idx = i + 1
                        break
            
            if end_idx != -1:
                json_content = content[start_idx:end_idx].strip()
                if json_content:
                    return json_content
    except Exception:
        pass
    
    # Method 4: Return the original content if it looks like JSON
    if content.startswith('{') and content.endswith('}'):
        return content
    if content.startswith('[') and content.endswith(']'):
        return content
    
    return ""


def parse_json_with_recovery(text: str) -> Dict:
    """Attempt to parse JSON, applying light sanitization if necessary."""
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        cleaned = sanitize_json_text(text)
        if cleaned and cleaned != text:
            try:
                return json.loads(cleaned)
            except json.JSONDecodeError:
                pass
        raise


def _format_percentage(numerator: int, denominator: int) -> float:
    if denominator <= 0:
        return 0.0
    return (numerator / denominator) * 100.0


def _describe_examples(example_indices: List[int]) -> str:
    if not example_indices:
        return ''
    if len(example_indices) == 1:
        return f" (occurred in round {example_indices[0]})"
    return f" (occurred in rounds {', '.join(map(str, example_indices))})"


def _collect_category_results(category_stats: Dict[str, Dict]) -> List[Dict]:
    results = []
    for key in ['in_box', 'attack', 'defense']:
        stats = category_stats.get(key, {})
        total = stats.get('count', stats.get('total', 0))
        wins = stats.get('wins', 0)
        rate = _format_percentage(wins, total)
        results.append({
            'key': key,
            'label': CATEGORY_LABELS.get(key, key),
            'definition': CATEGORY_DEFINITIONS.get(key, ''),
            'total': total,
            'wins': wins,
            'losses': max(total - wins, 0),
            'win_rate': rate
        })
    return results


def _build_style_and_risk_summary(
    category_results: List[Dict],
    metrics: Dict,
    loss_patterns: List[Dict]
) -> Tuple[str, str]:
    lines = ['Category explanation: In-Box = both sides initiate simultaneously; Attack = we strike first; Defense = opponent attacks first, we counter.']
    risks = []

    attack_result = next((c for c in category_results if c['key'] == 'attack'), None)
    defense_result = next((c for c in category_results if c['key'] == 'defense'), None)
    inbox_result = next((c for c in category_results if c['key'] == 'in_box'), None)

    total_attack = attack_result['total'] if attack_result else 0
    total_defense = defense_result['total'] if defense_result else 0
    total_inbox = inbox_result['total'] if inbox_result else 0

    if total_attack == 0 and (total_inbox > 0 or total_defense > 0):
        lines.append('Reactive fighting style: zero active attack samples, scoring mainly comes from in-box exchanges or defensive transitions.')
    elif total_inbox >= max(total_attack, total_defense) and inbox_result and inbox_result['win_rate'] >= 55:
        lines.append('Strong at initiative: in-box samples have high proportion and good win rate, can be considered primary scoring method, but this pattern requires validation for stability.')

    attack_effectiveness = sanitize_value(metrics.get('attack_effectiveness'))
    attack_distance_quality = sanitize_value(metrics.get('attack_distance_quality'))
    if attack_distance_quality >= 60 and attack_effectiveness <= 10:
        risks.append(
            'Attack distance judgment score is high but scoring rate is close to 0%, need to recalibrate attack "golden distance", beware of false confidence.'
        )

    if defense_result and defense_result['total'] > 0 and defense_result['win_rate'] <= 15:
        risks.append(
            f"Defense round win rate is only {defense_result['win_rate']:.1f}% ({defense_result['wins']}/{defense_result['total']}), indicating a systemic vulnerability."
        )
    elif defense_result and defense_result['total'] == 0:
        risks.append('Lack of defense samples, current defensive ability is unknown, need to supplement data through training or competition.')

    if attack_result and attack_result['total'] == 0:
        risks.append('No active attacks initiated in the match, need to confirm whether this is a technical shortcoming or tactical choice.')

    if inbox_result and inbox_result['total'] == 1 and inbox_result['win_rate'] == 100:
        risks.append('100% in-box win rate comes from single round only, need to validate stability of this action in training, avoid mistaking it as guaranteed winning tactic.')

    if not lines:
        lines.append('Fighting style not yet formed, need more samples to determine if style is proactive or reactive.')
    if not risks:
        risks.append('No significant risks currently, but continuous monitoring of win rates across different scenarios is still needed.')

    return '\n'.join(lines), '\n'.join(risks)


def _build_loss_summary(loss_patterns: List[Dict]) -> str:
    if not loss_patterns:
        return 'No significant loss patterns currently.'
    parts = []
    for pattern in loss_patterns:
        label = CATEGORY_LABELS.get(pattern['category'], pattern['category'])
        example = _describe_examples(pattern['examples'])
        parts.append(f"{label} - {pattern['reason_cn']}: {pattern['count']} times{example}")
    return '\n'.join(parts)


def _build_win_summary(win_patterns: List[Dict]) -> str:
    if not win_patterns:
        return 'No significant winning patterns currently.'
    parts = []
    for pattern in win_patterns:
        label = CATEGORY_LABELS.get(pattern['category'], pattern['category'])
        example = _describe_examples(pattern['examples'])
        parts.append(f"{label} - {pattern['reason_cn']}: {pattern['count']} times{example}")
    return '\n'.join(parts)


def _metric_line(label: str, value: float, suffix: str = '') -> str:
    if value is None:
        return f"{label}: No data"
    return f"{label}: {value:.1f}{suffix}"


def _build_metric_snapshot(metrics: Dict) -> str:
    lines = []
    pairs = [
        ('First Intention Scoring', sanitize_value(metrics.get('first_intention_effectiveness'))),
        ('Second Intention Scoring', sanitize_value(metrics.get('second_intention_effectiveness'))),
        ('Attack Promptness Index', sanitize_value(metrics.get('attack_promptness'))),
        ('Attack Distance Quality', sanitize_value(metrics.get('attack_distance_quality'))),
        ('Active Attack Scoring Rate', sanitize_value(metrics.get('attack_effectiveness'))),
        ('Defense Distance Management', sanitize_value(metrics.get('defense_distance_management'))),
        ('Counter Execution Rate', sanitize_value(metrics.get('counter_execution_rate'))),
        ('Defense Scoring Rate', sanitize_value(metrics.get('defense_effectiveness')))
    ]
    for label, value in pairs:
        lines.append(_metric_line(label, value, '%'))
    return '\n'.join(lines)


def _build_category_snapshot(category_results: List[Dict]) -> str:
    lines = []
    for result in category_results:
        if result['total'] == 0:
            lines.append(f"{result['label']}: 0 samples -> No data yet")
        else:
            qualifier = ' (very few samples)' if result['total'] < 3 else ''
            lines.append(
                f"{result['label']}: {result['wins']}/{result['total']} wins, win rate {result['win_rate']:.1f}%{qualifier}"
            )
    return '\n'.join(lines)


def _build_overall_prompt_context(
    fencer_side: str,
    total_touches: int,
    wins: int,
    losses: int,
    category_results: List[Dict],
    metrics: Dict,
    loss_patterns: List[Dict],
    win_patterns: List[Dict],
    loss_brief_lines: Optional[List[str]] = None,
    win_brief_lines: Optional[List[str]] = None
) -> Dict[str, str]:
    baseline = (
        f"Fencer: {SIDE_LABELS.get(fencer_side, fencer_side)}\n"
        f"Total Rounds: {total_touches}, Wins: {wins}, Losses: {losses}, Win Rate: {_format_percentage(wins, max(total_touches, 1)):.1f}%"
    )
    category_summary = _build_category_snapshot(category_results)
    metric_summary = _build_metric_snapshot(metrics)
    loss_summary = _build_loss_summary(loss_patterns)
    win_summary = _build_win_summary(win_patterns)

    if loss_brief_lines:
        extra_loss = '\n'.join(line for line in loss_brief_lines if line)
        if extra_loss:
            loss_summary = f"{loss_summary}\nIn-Depth Diagnosis:\n{extra_loss}"

    if win_brief_lines:
        extra_win = '\n'.join(line for line in win_brief_lines if line)
        if extra_win:
            win_summary = f"{win_summary}\nPattern Breakdown:\n{extra_win}"

    style_summary, risk_summary = _build_style_and_risk_summary(category_results, metrics, loss_patterns)

    return {
        'baseline_summary': baseline,
        'category_summary': category_summary,
        'metric_summary': metric_summary,
        'loss_summary': loss_summary,
        'win_summary': win_summary,
        'style_summary': style_summary,
        'risk_summary': risk_summary
    }


def _identify_strengths(category_results: List[Dict], metrics: Dict, best_category: Optional[Dict]) -> List[Dict]:
    strengths: List[Dict] = []
    attack = next((c for c in category_results if c['key'] == 'attack'), None)
    in_box = next((c for c in category_results if c['key'] == 'in_box'), None)

    if best_category and best_category['total'] > 0 and best_category['win_rate'] >= 55:
        strengths.append({
            'strength': f"{best_category['label']} is currently the most reliable scoring method",
            'evidence': (
                f"{best_category['label']} rounds total {best_category['total']} times, scored {best_category['wins']} times, win rate {best_category['win_rate']:.1f}%."
                f" Samples come from actual competition, indicating stability in decision-making and execution in this scenario."
            ),
            'enhancement_strategies': [
                f"Compile videos of these {best_category['wins']} successful rounds, annotate starting rhythm, footwork sequence, and hit locations to solidify as standard operations.",
                "Set up 'strongest combo rounds' in training, requiring replication of same starting and routes under match-paced rhythm, and record hit rate.",
                "Add [same setup - different finishes] exercises, for example adding direct thrust, line change thrust, continuous pressure as three endings after same opening, prevent being read by opponent."
            ]
        })

    attack_promptness = sanitize_value(metrics.get('attack_promptness'))
    attack_distance_quality = sanitize_value(metrics.get('attack_distance_quality'))
    attack_effectiveness = sanitize_value(metrics.get('attack_effectiveness'))

    if attack and attack['total'] > 0 and attack['win_rate'] >= 45:
        strengths.append({
            'strength': 'Active attack has scoring potential',
            'evidence': (
                f"Attack round win rate {attack['win_rate']:.1f}% ({attack['wins']}/{attack['total']}), "
                f"attack promptness index {attack_promptness:.1f}, distance quality {attack_distance_quality:.1f}."
            ),
            'enhancement_strategies': [
                'Break down active attack into three segments: "trigger condition - footwork sequence - attack line", and record execution for each round in training log.',
                'Use video to compare successful and failed active attacks, annotate starting timing, whether footwork is solid, and whether arm is pre-moving.',
                'Design "successful sample replication" training: extract moves from successful rounds, require replication at least 20 times under same rhythm and distance conditions.'
            ]
        })

    if in_box and in_box['total'] > 0 and in_box['win_rate'] >= 50:
        strengths.append({
            'strength': 'Close-range in-box has competitive edge',
            'evidence': (
                f"In-box round win rate {in_box['win_rate']:.1f}% ({in_box['wins']}/{in_box['total']}), "
                'indicating ability to maintain competitive results when both sides initiate at same time point.'
            ),
            'enhancement_strategies': [
                'Analyze successful in-box rounds frame-by-frame, quantify first-step starting timing, arm extension sequence, and final hit location.',
                'In in-box simulation, set rule "must complete one second action (remise or step-back counter) within two in-box exchanges", prevent being counter-hit by opponent.',
                'List corresponding in-box initiation plans for different opponent types (aggressive/defensive), and rotate usage in training.'
            ]
        })

    if not strengths:
        strengths.append({
            'strength': 'Basic foundation',
            'evidence': 'Current data shows no obvious highlights, overall structure still needs to be rebuilt.',
            'enhancement_strategies': [
                'Maintain core physical fitness and basic footwork training to ensure physical capability supports future improvements.',
                'Establish training log to track goal completion for each phase, providing basis for subsequent evaluation.',
                'Work with coach to set single-round tasks (e.g., attempt at least one scoring opportunity per round), gradually find advantages.'
            ]
        })

    return strengths[:3]


def _identify_weaknesses(
    category_results: List[Dict],
    metrics: Dict,
    loss_patterns: List[Dict]
) -> Tuple[List[Dict], str, List[Dict], Optional[Dict]]:
    weaknesses: List[Dict] = []
    priority_focus = 'Need to collect more data first to clarify training priorities.'

    patterns_by_category: Dict[str, List[Dict]] = defaultdict(list)
    for pattern in loss_patterns:
        patterns_by_category[pattern['category']].append(pattern)

    sorted_by_rate = sorted(
        [c for c in category_results if c['total'] > 0],
        key=lambda item: item['win_rate']
    )

    worst = sorted_by_rate[0] if sorted_by_rate else None
    if worst:
        if worst['total'] == 0:
            priority_focus = f"Primary task: supplement {worst['label']} samples to assess true performance."
        elif worst['win_rate'] <= 25:
            priority_focus = f"Primary task: rebuild {worst['label']} ({worst['win_rate']:.1f}% win rate)."

    defense_result = next((c for c in category_results if c['key'] == 'defense'), None)
    attack_result = next((c for c in category_results if c['key'] == 'attack'), None)
    inbox_result = next((c for c in category_results if c['key'] == 'in_box'), None)

    defense_distance = sanitize_value(metrics.get('defense_distance_management'))
    counter_exec = sanitize_value(metrics.get('counter_execution_rate'))
    defense_effectiveness = sanitize_value(metrics.get('defense_effectiveness'))
    attack_promptness = sanitize_value(metrics.get('attack_promptness'))
    attack_distance_quality = sanitize_value(metrics.get('attack_distance_quality'))
    attack_effectiveness = sanitize_value(metrics.get('attack_effectiveness'))

    if defense_result and defense_result['total'] > 0 and defense_result['win_rate'] <= 25:
        reason = patterns_by_category.get('defense', [])
        leading_reason = reason[0] if reason else None
        impact_parts = [
            f"Defense rounds {defense_result['total']} times, only {defense_result['wins']} times scored (win rate {defense_result['win_rate']:.1f}%).",
            f"Defense distance management score {defense_distance:.1f}, counter conversion rate {counter_exec:.1f}%, defense scoring index {defense_effectiveness:.1f}."
        ]
        improvement = [
            'Establish "retreat two steps - stabilize - parry" defensive baseline loop, complete 30 sets per training session, ensure attack-right awareness maintained while retreating.',
            'Specifically train first-beat riposte after parry-4/6, do 10 continuous sparring sessions, require at least 4 successes.',
            'In high-pressure combat, set rule "defense block without counter requires extra practice", force counter-attack consciousness.'
        ]
        if leading_reason:
            impact_parts.append(
                f"Main error: {leading_reason['reason_cn']}, total {leading_reason['count']} times{_describe_examples(leading_reason['examples'])}."
            )
            improvement.extend(LOSS_REASON_REMEDIATIONS.get(leading_reason['reason_key'], []))
        weaknesses.append({
            'weakness': 'Defense system missing, almost unable to score through defense',
            'impact': ' '.join(impact_parts),
            'improvement_strategies': improvement[:5]
        })
    elif defense_result and defense_result['total'] == 0:
        weaknesses.append({
            'weakness': 'Lack of defensive counter samples, unable to evaluate defensive capability',
            'impact': 'Current match did not show effective defensive scoring rounds, need to collect data through training simulation.',
            'improvement_strategies': [
                'Arrange at least 20 "retreat - parry - riposte" scenarios in training sessions, and record success rate.',
                'Use video playback to confirm frame-by-frame whether distance control and parry actions are standard during retreat.',
                'In team sparring, set scoring rules: successful defense without counter requires additional training, encourage initiative.'
            ]
        })

    attack_promptness = sanitize_value(metrics.get('attack_promptness'))
    attack_effectiveness = sanitize_value(metrics.get('attack_effectiveness'))
    first_intention = sanitize_value(metrics.get('first_intention_effectiveness'))
    second_intention = sanitize_value(metrics.get('second_intention_effectiveness'))

    if attack_result and attack_result['total'] > 0 and attack_result['win_rate'] < 45:
        reason = patterns_by_category.get('attack', [])
        leading_reason = reason[0] if reason else None
        impact_parts = [
            f"Active attack scoring rate only {attack_result['win_rate']:.1f}% ({attack_result['wins']}/{attack_result['total']}).",
            f"First intention scoring {first_intention:.1f}%, second intention {second_intention:.1f}%, indicating lack of sustained threat after initiation."
        ]
        improvement = [
            'Use "feint + real attack" combination training (3 beats per round), specifically practice line changes for frequently-judged lines.',
            'Immediately set up "second beat takeover" practice after attack, require either retreat or continuous remise within 0.2 seconds after lunge landing.',
            'Analyze blocked attack frames through video, count whether it\'s distance too far, rhythm too slow, or line anticipated, and modify actions targeting the problem.'
        ]
        if leading_reason:
            impact_parts.append(
                f"Common problem: {leading_reason['reason_cn']}, occurred {leading_reason['count']} times{_describe_examples(leading_reason['examples'])}."
            )
            improvement.extend(LOSS_REASON_REMEDIATIONS.get(leading_reason['reason_key'], []))
        weaknesses.append({
            'weakness': 'Active attack output unstable',
            'impact': ' '.join(impact_parts),
            'improvement_strategies': improvement[:5]
        })
    elif attack_result and attack_result['total'] == 0:
        weaknesses.append({
            'weakness': 'Lack of active attack attempts in match',
            'impact': 'Lack of active initiative will lead to overly passive strategy, once in-box fails there is no way to turn the tide.',
            'improvement_strategies': [
                'Set rule in daily training "must design one active attack per round", gradually establish active consciousness.',
                'Plan two sets of active opening combos with coach, and test success rate in small-score sparring.',
                'Use virtual opponent or video feedback practice, execute completely from preparation to execution.'
            ]
        })

    if (
        attack_result and attack_result['total'] > 0 and
        attack_distance_quality >= 70 and
        attack_effectiveness <= 10
    ):
        weaknesses.append({
            'weakness': 'Attack distance indicator seriously disconnected from scoring results',
            'impact': (
                f"Attack distance quality score {attack_distance_quality:.1f}, but active attack scoring rate only {attack_effectiveness:.1f}%. "
                'Indicates current "golden distance" judgment has systemic error, rashly attacking may result in being counter-attacked.'
            ),
            'improvement_strategies': [
                'Review all active attack videos round by round, mark actions within two beats after entering distance, and record specific reasons for losing points.',
                'Set distance markers in training ground, have coach randomly call out "continue pressing/immediate strike/feint retreat", re-establish mapping between distance and decisions.',
                'Arrange 20 "enter distance → complete attack within two beats" sparring sessions, count actual hit rate, revise distance indicator thresholds accordingly.',
                'Record new distance judgment rules in tactical manual, and complete one tracking validation before next match.'
            ]
        })

    if inbox_result and inbox_result['total'] > 0 and inbox_result['win_rate'] < 45:
        reason = patterns_by_category.get('in_box', [])
        leading_reason = reason[0] if reason else None
        impact_parts = [
            f"In-box round win rate {inbox_result['win_rate']:.1f}%, falling behind in initiative phase.",
            'In-box hesitation will directly let opponent seize initiative.'
        ]
        improvement = [
            'Conduct "steady start" vs "explosive start" comparison practice, shorten first beat reaction time and record frames.',
            'In in-box simulation, set "must complete one second action (remise or step-back counter) within two in-box exchanges", require coach to score immediately.',
            'Combine with coach hand target training, improve instant judgment ability through different initiation signals or visual signals.'
        ]
        if leading_reason:
            impact_parts.append(
                f"Typical error: {leading_reason['reason_cn']} ({leading_reason['count']} times){_describe_examples(leading_reason['examples'])}."
            )
            improvement.extend(LOSS_REASON_REMEDIATIONS.get(leading_reason['reason_key'], []))
        weaknesses.append({
            'weakness': 'In-box phase slow initiation/decision hesitation',
            'impact': ' '.join(impact_parts),
            'improvement_strategies': improvement[:5]
        })
    elif inbox_result and inbox_result['total'] == 0:
        weaknesses.append({
            'weakness': 'Lack of in-box samples, unable to judge initiative ability',
            'impact': 'Once forced into simultaneous initiation scenario in match, may be completely passive.',
            'improvement_strategies': [
                'Arrange "forced in-box" training: coach orders both sides to start simultaneously, record success rate and strike time.',
                'Through opening simulation practice, immediately enter in-box after pointing blade at opponent, cultivate initiative courage.',
                'Collect in-box videos from actual combat or sparring, analyze frame-by-frame initiation signals and hand-foot coordination.'
            ]
        })

    if not weaknesses:
        weaknesses.append({
            'weakness': 'Insufficient data to pinpoint core problems',
            'impact': 'Current sample size is limited, suggest supplementing more match data to identify systemic vulnerabilities.',
            'improvement_strategies': [
                'Arrange more actual combat rounds and record category data, accumulate at least 10 valid samples.',
                'Fill in round record sheet after each training session, annotate results of attack/in-box/defense.',
                'Before data is sufficient, maintain balanced basic technical training, avoid imbalance.'
            ]
        })

    return weaknesses[:4], priority_focus, sorted_by_rate, worst


def _build_situational_strategies(
    category_results: List[Dict],
    best_category: Optional[Dict],
    worst_category: Optional[Dict],
    priority_focus: str
) -> List[Dict]:
    strategies: List[Dict] = []

    if best_category and best_category['total'] > 0:
        sample_note = ''
        if best_category['total'] < 6:
            sample_note = f" ({best_category['total']} samples, need to stress-test stability in training again)"

        best_key = best_category.get('key')
        if best_key == 'defense':
            tactics = [
                f"Through footwork rhythm and feints, induce opponent to move first, creating {best_category['label']} window{sample_note}.",
                f"Pre-set parry-riposte sequence, maintain distance buffer during retreat, to leverage current {best_category['label']} win rate {best_category['win_rate']:.1f}% advantage.",
                'If opponent hesitates to attack, actively press forward beyond their comfort zone to force their hand, then complete scoring with established riposte line.'
            ]
        elif best_key == 'in_box':
            tactics = [
                f"Actively create close-range initiative rhythm{sample_note}, use verified in-box combos to seize initiative.",
                'Through rhythm breaks or anticipating initiation points, force opponent to start simultaneously, then execute highest hit-rate combination.',
                'Remember to immediately follow up with next beat after in-box success, prevent opponent counter due to hesitation.'
            ]
        else:
            tactics = [
                f"Accelerate advancement and prioritize using highest success rate {best_category['label']} combo{sample_note}.",
                'Design in advance "if first beat is blocked" follow-up action, avoid opponent counter-attacking to seize back rhythm.',
                'Through continuous pressure, force opponent to retreat, creating space for launching same combo again.'
            ]

        strategies.append({
            'scenario': 'When score is behind',
            'tactics': tactics
        })
    else:
        strategies.append({
            'scenario': 'When score is behind',
            'tactics': [
                'Avoid passive waiting, treat each round as active initiative opportunity, strive to regain initiative first.',
                'Use pre-set fast-paced attack combinations to quickly change rhythm.',
                'If defensive scoring ability is weak, need to reduce opponent-dominated rounds through continuous pressure.'
            ]
        })

    balanced = next((c for c in category_results if c['total'] > 0 and 40 <= c['win_rate'] <= 60), None)
    strategies.append({
        'scenario': 'When leading or evenly matched',
        'tactics': [
            'Use "probe - feint - real attack" loop, steadily expand advantage.',
            'Through distance control, force opponent to move first, prepare counter-plan at medium distance.',
            'Regularly switch rhythm, prevent opponent from adapting to single style.'
        ]
    })

    defense_result = next((c for c in category_results if c['key'] == 'defense'), None)
    if defense_result and defense_result['total'] > 0 and defense_result['win_rate'] <= 25:
        strategies.append({
            'scenario': 'Facing aggressive opponent',
            'tactics': [
                'Plan retreat path in advance, avoid being dragged to back line before being forced to defend.',
                'Set "retreat two steps then counter" trigger condition, reduce pure passive being-hit time.',
                'If defense still cannot score, decisively switch to counter-pressure, regain rhythm through initiative.'
            ]
        })
    else:
        strategies.append({
            'scenario': 'Facing aggressive opponent',
            'tactics': [
                'Use distance control and rhythm changes to resolve opponent\'s sustained pressure.',
                'Observe opponent\'s pre-attack signs, pre-arrange parry-riposte or sidestep counter.',
                'If opponent habitually charges single line, use feints to induce misstep then change line attack.'
            ]
        })

    attack_result = next((c for c in category_results if c['key'] == 'attack'), None)
    if attack_result and attack_result['total'] > 0 and attack_result['win_rate'] < 45:
        strategies.append({
            'scenario': 'Opponent skilled at waiting for counter',
            'tactics': [
                'Use probing steps and feints to force opponent to prematurely expose defensive lines.',
                'Reduce frontal direct attacks, switch to mobilizing attacks or second-intention attacks.',
                'When opponent prepares to counter, prepare line changes or pauses in advance, disrupt their rhythm.'
            ]
        })

    return strategies[:4]


def _build_development_roadmap(
    priority_focus: str,
    category_results: List[Dict],
    best_category: Optional[Dict],
    worst_category: Optional[Dict]
) -> List[Dict]:
    roadmap: List[Dict] = []

    short_focus = priority_focus.replace('Primary task: ', '').replace('首要任务：', '')
    roadmap.append({
        'phase': 'Short-term (1-2 weeks)',
        'focus': short_focus,
        'methods': [
            'Set quantitative indicators for training sessions (hit rate, riposte count, etc.), record and review each training session.',
            'Arrange weekly video review session, compare frame-by-frame whether improvements are reflected in action details.',
            'Use small-score actual combat (5-point system) to validate short-cycle training results, and record success/failure reasons.'
        ]
    })

    if worst_category and worst_category['total'] > 0:
        mid_focus = f"Validate whether {worst_category['label']} remedial training is effective"
    else:
        mid_focus = 'Validate existing advantages and establish second scoring dimension'
    mid_methods = [
        'For main weakness scenarios, arrange at least one high-repetition drill per week, record hit and error reasons.',
        'In sparring, simulate different style opponents (aggressive, defensive, counter), observe whether advantages are stable.',
        'Set phase indicators (e.g., defensive counter training match scoring rate ≥15%), have coach verify weekly.'
    ]
    roadmap.append({
        'phase': 'Medium-term (1-2 months)',
        'focus': mid_focus,
        'methods': mid_methods
    })

    long_focus = 'Form complete attack-defense system executable in important matches'
    long_methods = [
        'Test new tactical combinations in high-intensity match simulation, confirm still executable under pressure conditions.',
        'Re-test data once per month, confirm weaknesses are addressed and adjust training indicators timely.',
        'Arrange comprehensive training days for physical, psychological, and technical aspects, ensure advantage items remain threatening under fatigue and pressure.'
    ]
    roadmap.append({
        'phase': 'Long-term (3+ months)',
        'focus': long_focus,
        'methods': long_methods
    })

    return roadmap


def _build_overall_performance_summary(
    fencer_side: str,
    total_touches: int,
    wins: int,
    category_stats: Dict[str, Dict],
    performance_metrics: Dict,
    loss_analysis: Dict,
    win_analysis: Dict,
    match_data: List[Dict]
) -> Dict:
    losses = max(total_touches - wins, 0)
    win_rate = _format_percentage(wins, total_touches)
    category_results = _collect_category_results(category_stats)
    loss_patterns = _extract_top_loss_patterns(loss_analysis, fencer_side, max_entries=5)
    win_patterns = _extract_top_win_patterns(win_analysis, fencer_side, max_entries=5)

    side_label = SIDE_LABELS.get(fencer_side, fencer_side)

    profile_lines = [
        f"{side_label} participated in {total_touches} rounds, wins {wins}, losses {losses}, overall win rate {win_rate:.1f}%.",
        'Category explanation: system divides rounds by action dominance — ' + '; '.join(
            f"{CATEGORY_LABELS[k]}={CATEGORY_DEFINITIONS[k]}" for k in CATEGORY_LABELS
        )
    ]

    for result in category_results:
        if result['total'] == 0:
            profile_lines.append(f"{result['label']}: No samples")
        else:
            profile_lines.append(
                f"{result['label']}: {result['wins']}/{result['total']} wins, win rate {result['win_rate']:.1f}%"
            )

    if loss_patterns:
        top_issue = loss_patterns[0]
        profile_lines.append(
            f"Main loss pattern: {CATEGORY_LABELS.get(top_issue['category'], top_issue['category'])} phase showed "
            f"\"{top_issue['reason_cn']}\" {top_issue['count']} times{_describe_examples(top_issue['examples'])}."
        )
    if win_patterns:
        top_win = win_patterns[0]
        profile_lines.append(
            f"Main scoring pattern: {CATEGORY_LABELS.get(top_win['category'], top_win['category'])} phase relied on "
            f"\"{top_win['reason_cn']}\" {top_win['count']} times{_describe_examples(top_win['examples'])}."
        )

    performance_profile = '<br>'.join(profile_lines)

    best_category = None
    if category_results:
        eligible = [c for c in category_results if c['total'] >= 3]
        if eligible:
            best_category = max(eligible, key=lambda item: item['win_rate'])

    strengths = _identify_strengths(category_results, performance_metrics, best_category)
    weaknesses, priority_focus, category_order, worst_category = _identify_weaknesses(
        category_results,
        performance_metrics,
        loss_patterns
    )
    situational_strategies = _build_situational_strategies(category_results, best_category, worst_category, priority_focus)
    roadmap = _build_development_roadmap(priority_focus, category_results, best_category, worst_category)

    total_touches = max(total_touches, 0)
    if total_touches < 4:
        readiness = f"Only {total_touches} round samples, suggest accumulating more actual combat data before assessing competitive status."
    elif worst_category and worst_category['total'] > 0 and worst_category['win_rate'] <= 25:
        readiness = (
            f"{worst_category['label']} win rate only {worst_category['win_rate']:.1f}% ({worst_category['wins']}/{worst_category['total']}), "
            "must prioritize fixing this scenario before participating in high-intensity events."
        )
        readiness = ''.join(readiness)
    elif best_category and best_category['win_rate'] >= 60 and (not worst_category or worst_category['win_rate'] >= 40):
        readiness = (
            f"Core scoring point is {best_category['label']} (win rate {best_category['win_rate']:.1f}%), "
            "attack-defense structure relatively balanced, can enter high-intensity events but still need to maintain advantages."
        )
        readiness = ''.join(readiness)
    else:
        readiness = "Attack-defense structure not yet stable, suggest completing focused strengthening of key scenarios in training before arranging critical events."

    if worst_category:
        if worst_category['total'] == 0:
            immediate_focus = f"Supplement {worst_category['label']} match and training samples, establish baseline data."
        else:
            immediate_focus = f"Develop specialized improvement plan for {worst_category['label']} (win rate {worst_category['win_rate']:.1f}%)."
    else:
        immediate_focus = priority_focus.replace('Primary task: ', '').replace('首要任务：', '')

    key_insights = []
    for result in category_results:
        if result['total'] == 0:
            key_insights.append(f"{result['label']} samples insufficient, need to deliberately increase training and match data for related scenarios.")
        elif result['total'] < 3:
            key_insights.append(
                f"{result['label']} samples only {result['total']} times, current win rate {result['win_rate']:.1f}% is for reference only, must verify through training or matches."
            )
        elif result['win_rate'] <= 15:
            key_insights.append(
                f"{result['label']} win rate only {result['win_rate']:.1f}%, is a systemic weak point, must be made the core of phase training."
            )
        elif result['win_rate'] >= 55:
            key_insights.append(
                f"{result['label']} maintains {result['win_rate']:.1f}% win rate, can serve as main scoring method for current match phase."
            )
    if not key_insights:
        key_insights.append('Data samples limited, need to continuously accumulate and re-test indicators in subsequent training/matches.')

    return {
        'performance_profile': performance_profile,
        'detailed_strengths': strengths,
        'critical_weaknesses': weaknesses,
        'situational_strategies': situational_strategies,
        'development_roadmap': roadmap,
        'competition_readiness': readiness,
        'immediate_focus': immediate_focus,
        'key_insights': key_insights
    }


def _apply_overall_post_processing(
    output: Dict,
    category_results: List[Dict],
    total_touches: int,
    loss_patterns: List[Dict],
    win_patterns: List[Dict],
    performance_metrics: Dict
) -> Dict:
    key_insights = output.setdefault('key_insights', [])
    seen = set(key_insights)

    def _add_insight(message: str, *, priority: bool = False) -> None:
        if not message:
            return
        if message in seen:
            return
        if priority:
            key_insights.insert(0, message)
        else:
            key_insights.append(message)
        seen.add(message)

    if total_touches < 4:
        _add_insight(f"Overall samples only {total_touches} rounds, need to verify conclusions in subsequent matches.", priority=True)

    low_sample_notes = []
    for result in category_results:
        if result['total'] == 0:
            low_sample_notes.append(f"{result['label']} no samples")
        elif result['total'] < 3:
            low_sample_notes.append(f"{result['label']} only {result['total']} times (win rate {result['win_rate']:.1f}%)")
    if low_sample_notes:
        _add_insight("Sample notes: " + "; ".join(low_sample_notes))

    attack_distance_quality = sanitize_value(performance_metrics.get('attack_distance_quality'))
    attack_effectiveness = sanitize_value(performance_metrics.get('attack_effectiveness'))
    if attack_distance_quality >= 70 and attack_effectiveness <= 10:
        _add_insight(
            f"Active attack distance score {attack_distance_quality:.1f} but scoring rate only {attack_effectiveness:.1f}%, need to calibrate finishing actions and distance selection."
        )

    for pattern in loss_patterns or []:
        if pattern['count'] >= 3:
            _add_insight(
                f"{CATEGORY_LABELS.get(pattern['category'], pattern['category'])} repeatedly showed \"{pattern['reason_cn']}\" ({pattern['count']} times), should be listed in focused improvement list."
            )

    for pattern in win_patterns or []:
        if pattern['count'] >= 3:
            _add_insight(
                f"{CATEGORY_LABELS.get(pattern['category'], pattern['category'])} stably scored with \"{pattern['reason_cn']}\" {pattern['count']} times, can serve as current main attack combo."
            )

    max_insights = 5
    if len(key_insights) > max_insights:
        output['key_insights'] = key_insights[:max_insights]
    else:
        output['key_insights'] = key_insights
    return output


def _enforce_global_rate_limit() -> None:
    global _GEMINI_NEXT_AVAILABLE_TS
    now = time.time()
    if now < _GEMINI_NEXT_AVAILABLE_TS:
        delay = max(0.0, _GEMINI_NEXT_AVAILABLE_TS - now)
        time.sleep(delay)
    _GEMINI_NEXT_AVAILABLE_TS = time.time() + GEMINI_MIN_INTERVAL_S


def _sleep_with_jitter(base_seconds: float) -> None:
    jitter = base_seconds * 0.25 * random.random()
    time.sleep(base_seconds + jitter)


def _apply_rate_penalty(penalty_seconds: float, reason: str = '') -> None:
    """Push next-available window forward to absorb API throttling."""
    global _GEMINI_NEXT_AVAILABLE_TS
    penalty_seconds = max(penalty_seconds, GEMINI_MIN_INTERVAL_S)
    next_slot = time.time() + penalty_seconds
    if next_slot > _GEMINI_NEXT_AVAILABLE_TS:
        _GEMINI_NEXT_AVAILABLE_TS = next_slot
    if reason:
        logging.debug(f"Applied Gemini rate penalty ({penalty_seconds:.2f}s) due to {reason}")


def call_gemini_api(prompt: str, json_data: Dict, max_retries: int = 3) -> Dict:
    """Call Gemini API to analyze loss reasons for a touch"""
    
    # Try to get API key from environment variable
    api_key = os.getenv('GEMINI_API_KEY')
    if not api_key:
        logging.warning("GEMINI_API_KEY not found in environment variables")
        return {
            '__parse_error__': 'missing_api_key',
            'error': 'Gemini API key not configured'
        }
    
    # Prepare the full prompt with JSON data
    full_prompt = prompt.replace('{JSON_DATA}', json.dumps(json_data, indent=2))
    
    model_name = GEMINI_MODEL or 'gemini-2.5-flash-lite'
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{model_name}:generateContent?key={api_key}"
    
    headers = {
        'Content-Type': 'application/json',
    }

    generation_config = {
        "temperature": 0.1,
        "topK": 1,
        "topP": 0.8,
        "maxOutputTokens": 8192,
        "responseMimeType": "application/json"
    }

    payload = {
        "contents": [{
            "role": "user",
            "parts": [{
                "text": full_prompt
            }]
        }],
        "generationConfig": generation_config
    }
    
    prompt_lower = prompt.lower()
    expects_overall = 'performance_profile' in prompt_lower or 'overall_analysis' in prompt_lower
    expects_category = 'performance_summary' in prompt_lower or 'category' in prompt_lower

    last_error: Optional[Exception] = None

    for attempt in range(max_retries):
        try:
            _enforce_global_rate_limit()
            response = requests.post(url, json=payload, headers=headers, timeout=180)

            status_code = response.status_code
            if status_code in (429, 503):
                reason = response.reason or 'Rate limited'
                logging.warning(f"Gemini API throttled request (status {status_code}): {reason}")
                penalty = min(GEMINI_BACKOFF_MAX_S, GEMINI_MIN_INTERVAL_S * (6 if status_code == 503 else 4))
                _apply_rate_penalty(penalty, f'HTTP {status_code}')
                last_error = requests.exceptions.HTTPError(f"{status_code} {reason}", response=response)
                if attempt < max_retries - 1:
                    continue
                break

            response.raise_for_status()

            result = response.json()

            if 'error' in result:
                error_info = result.get('error', {})
                message = error_info.get('message', 'Gemini API error response')
                code_raw = error_info.get('code')
                try:
                    code = int(code_raw)
                except (TypeError, ValueError):
                    code = code_raw
                logging.error(f"Gemini API returned error payload (code={code_raw}): {message}")
                if isinstance(code, int) and code in (429, 503) and attempt < max_retries - 1:
                    penalty = min(GEMINI_BACKOFF_MAX_S, GEMINI_MIN_INTERVAL_S * (6 if code == 503 else 4))
                    _apply_rate_penalty(penalty, f'payload error {code}')
                    continue
                last_error = RuntimeError(message)
                break

            candidates = result.get('candidates') or []
            if not candidates:
                logging.error("No candidates in Gemini API response")
                last_error = RuntimeError('no_candidates')
                break

            text_parts: List[str] = []
            json_fragments = []
            for candidate in candidates:
                parts = []
                content_obj = candidate.get('content')
                if isinstance(content_obj, dict):
                    parts = content_obj.get('parts') or []
                elif isinstance(content_obj, list):
                    parts = content_obj

                for part in parts:
                    if not isinstance(part, dict):
                        continue
                    if 'jsonValue' in part and part['jsonValue'] is not None:
                        json_fragments.append(part['jsonValue'])
                    if 'text' in part and isinstance(part['text'], str):
                        text_parts.append(part['text'])

                legacy_text = candidate.get('text') or candidate.get('output')
                if isinstance(legacy_text, str) and legacy_text.strip():
                    text_parts.append(legacy_text)

            if not text_parts and not json_fragments:
                # Fallback: sometimes `candidate` may include `output` field or `text`
                for candidate in candidates:
                    text_val = candidate.get('text') or candidate.get('output')
                    if isinstance(text_val, str) and text_val.strip():
                        text_parts.append(text_val)

            if json_fragments:
                candidate_json: Any
                candidate_json = json_fragments[0]
                if isinstance(candidate_json, str):
                    try:
                        candidate_json = json.loads(candidate_json)
                    except Exception:
                        logging.warning("jsonValue returned string that is not valid JSON; using raw string")
                logging.info("Gemini returned structured JSON via jsonValue; using it directly")
                return candidate_json

            combined_text = ''.join(text_parts)
            combined_text = combined_text.strip()

            if not combined_text:
                logging.error("Gemini response missing textual content")
                last_error = RuntimeError('empty_response')
                break

            logging.info(f"Gemini API response (first 300 chars): {combined_text[:300]}...")

            json_content = extract_json_from_markdown(combined_text)
            try:
                if json_content:
                    logging.debug(f"Extracted JSON content (first 200 chars): {json_content[:200]}...")
                    parsed_result = parse_json_with_recovery(json_content)
                    logging.info("Successfully parsed JSON from Gemini response")
                    return parsed_result

                parsed_result = parse_json_with_recovery(combined_text)
                logging.info("Successfully parsed original content as JSON")
                return parsed_result

            except (json.JSONDecodeError, ValueError, IndexError, UnicodeDecodeError) as e:
                logging.error(f"Failed to parse Gemini response as JSON: {e}")
                logging.error(f"Raw content (first 500 chars): {combined_text[:500]}...")
                if json_content:
                    logging.error(f"Extracted content that failed to parse: {json_content[:300]}...")
                logging.error("JSON extraction methods attempted: markdown code blocks, bracket counting, raw parsing")

                last_error = e
                if attempt < max_retries - 1:
                    logging.info("Retrying Gemini request due to JSON parse failure")
                    time.sleep(0.6)
                    continue
                break

        except requests.exceptions.RequestException as e:
            status_code = getattr(getattr(e, 'response', None), 'status_code', None)
            logging.error(f"Gemini API request failed (attempt {attempt + 1}): {e}")
            last_error = e
            if status_code in (429, 503):
                penalty = min(GEMINI_BACKOFF_MAX_S, GEMINI_MIN_INTERVAL_S * (6 if status_code == 503 else 4))
                _apply_rate_penalty(penalty, f'exception {status_code}')
            if attempt < max_retries - 1:
                # Exponential backoff with jitter, capped
                backoff = min(GEMINI_BACKOFF_MAX_S, GEMINI_BACKOFF_BASE_S * (2 ** attempt))
                _sleep_with_jitter(backoff)
                continue
            break

    error_detail = str(last_error) if last_error else 'parse_failure'

    if expects_category:
        return {
            '__parse_error__': True,
            'performance_summary': 'API response parsing error',
            'technical_analysis': 'JSON parsing failed',
            'tactical_analysis': 'JSON parsing failed',
            'recurring_problems': [],
            'detailed_improvements': [],
            'opponent_exploitation': [],
            'competition_strategies': [],
            'priority_training_plan': [],
            'overall_rating': '6',
            'key_insights': ['Gemini output did not generate valid JSON'],
            'error_detail': error_detail
        }
    if expects_overall:
        return {
            '__parse_error__': True,
            'performance_profile': 'API response parsing error',
            'detailed_strengths': [],
            'critical_weaknesses': [],
            'situational_strategies': [],
            'development_roadmap': [],
            'competition_readiness': 'Assessment unavailable',
            'immediate_focus': 'Awaiting valid analysis',
            'key_insights': ['Gemini output did not generate valid JSON'],
            'error_detail': error_detail
        }
    return {
        '__parse_error__': True,
        'loss_category': 'API Request Error',
        'loss_sub_category': 'Max Retries Exceeded',
        'brief_reasoning': 'Exceeded maximum retries for Gemini API',
        'error_detail': error_detail
    }

def get_loss_analysis_prompts() -> Dict[str, str]:
    """Return the prompts for each category loss analysis"""
    
    prompts = {
        'in_box': '''You are an expert AI fencing analyst named "Coach Sabre." Your specialty is diagnosing tactical and physical errors from performance data.

Your task is to analyze the provided JSON data for a single fencing touch where a fencer lost during the **In-Box** phase (the initial fight for initiative). This is not a simple logical checklist. You must perform a holistic analysis, weigh all the available data clues, and use your expert judgment to determine the **single, most significant reason** for the loss from the classification system below. 

---

### **In-Box Loss Classification System**

*   **Sub-category 1.1: Slow Reaction at Start**
    *   *Reasoning:* The fencer was significantly slower to react to the "Allez" call, giving the opponent a critical head start.
    *   *Data Clues:* The primary evidence is a much higher `'init_time'` in the loser's `'first_step'` metrics compared to the winner.

*   **Sub-category 1.2: Outmatched by Speed & Power**
    *   *Reasoning:* The fencer started on time but was physically outmatched. The opponent's first step was significantly faster, more explosive, or more powerful, allowing them to dominate the initial space.
    *   *Data Clues:* Look for similar `'init_time'` but significantly lower `'velocity'` or `'acceleration'` in the loser's `'first_step'` metrics. This is about being physically overpowered, not late.

*   **Sub-category 1.3: Indecisive Movement / Early Pause** 
    *   *Reasoning:* The fencer hesitated in the opening moments. An early pause or break in rhythm destroyed their forward momentum, creating a perfect window for the opponent to attack into their inaction.
    *   *Data Clues:* Look for a `'pause'` interval for the loser that starts very early (e.g., within the first second), especially while the winner is in an `'advance'` interval.

*   **Sub-category 1.4: Lack of Lunging** 
    *   *Reasoning:* The fencer lost the initial exchange because they failed to present a credible offensive threat. By not attempting a lunge early, they allowed the opponent to advance and score without any pressure.
    *   *Data Clues:* The key evidence is 'has_launch': false` for the loser within the first 1-1.5 seconds.

*   **Sub-category 1.5: Lack of Arm Extension**
    *   *Reasoning:* The fencer lost the initial exchange because they failed to present a credible offensive threat. By not extending their arm, they allowed the opponent to advance and score without any pressure.
    *   *Data Clues:* The key evidence is 'arm_extension_freq': 0` for the loser within the first 1-1.5 seconds.

---

### **Your Task**

Analyze the following JSON data. Identify the loser, confirm their category is "in_box," and select the most fitting sub-category from the system above.
Before reaching a final conclusion, evaluate each candidate sub-category one by one, explain whether it fits the data, then select the most appropriate one.

**DATA:**
```json
{JSON_DATA}
```

**OUTPUT FORMAT:**
Your response MUST be a single, valid JSON object and nothing else. Use the following structure:
```json
{
  "loss_category": "In-Box",
  "loss_sub_category": "The name of the sub-category you selected",
  "brief_reasoning": "Explain in English why this option fits best, indicate why other options do not hold, and cite core data."
}
```''',

        'attack': '''You are an expert AI fencing analyst named "Coach Sabre." Your specialty is diagnosing tactical and physical errors from performance data.

Your task is to analyze the provided JSON data for a single fencing touch where a fencer lost while **Attacking**. This is not a simple logical checklist. You must perform a holistic analysis, weigh all the available data clues, and use your expert judgment to determine the **single, most significant reason** for the loss from the classification system below.

---

### **Attack Loss Classification System**

*   **Sub-category 2.1: Attacked from Too Far (Positional Error)** 
    *   *Reasoning:* The attack was launched from outside the optimal hitting distance, giving the opponent too much time to prepare a successful defense or counter.
    *   *Data Clues:* The most direct evidence is `'good_attack_distance': false` in the loser's `'interval_analysis.advance_analyses'` for the attacking interval.

*   **Sub-category 2.2: Predictable Attack (Tactical Error)**
    *   *Reasoning:* The attack was executed with a simple action and a steady, constant rhythm, making it predictable and easy for the opponent to time a defensive response.
    *   *Data Clues:* Look for a combination of `'tempo_type': 'steady_tempo'` and `'attack_info.attack_type': 'simple_attack'` in the loser's attack analysis.

*   **Sub-category 2.3: Countered on Preparation (Timing Error)**
    *   *Reasoning:* The fencer began building their attack, but the opponent timed a faster, more decisive attack *into* their preparation, stealing the initiative before the attack could be completed.
    *   *Data Clues:* This is a complex interaction. The strongest evidence is an overlap in time between the loser's `'advance'` interval and the winner's `'advance'` interval, where the winner's advance starts *later* but their `'attacking_acceleration'` is significantly higher. This implies a powerful, reactive attack.

*   **Sub-category 2.4: Passive/Weak Attack (Execution Failure)** 
    *   *Reasoning:* The offensive action lacked the necessary speed and power to be a credible threat, allowing the opponent to easily ignore it and launch a successful counter-action.
    *   *Data Clues:* Look for very low `'attacking_velocity'` and `'attacking_acceleration'` for the loser, indicating a lack of conviction.

---

### **Your Task**

Analyze the following JSON data. Identify the loser, confirm their category is "attack," and select the most fitting sub-category from the system above. First evaluate each sub-category one by one to determine if it fits the data, then select the best conclusion.

**DATA:**
```json
{JSON_DATA}
```

**OUTPUT FORMAT:**
Your response MUST be a single, valid JSON object and nothing else. Use the following structure:
```json
{
  "loss_category": "Attack",
  "loss_sub_category": "The name of the sub-category you selected",
  "brief_reasoning": "Explain core cause in English, briefly describe why other options were excluded, cite key data points."
}
```''',

        'defense': '''You are an expert AI fencing analyst named "Coach Sabre." Your specialty is diagnosing tactical and physical errors from performance data.

Your task is to analyze the provided JSON data for a single fencing touch where a fencer lost while **Defending**. This is not a simple logical checklist. You must perform a holistic analysis, weigh all the available data clues, and use your expert judgment to determine the **single, most significant reason** for the loss from the classification system below. 

---

### **Defense Loss Classification System**

*   **Sub-category 3.1: Collapsed Distance (Positional Error)** 
    *   *Reasoning:* While retreating, the fencer allowed the opponent to get too close, which eliminated their own time and space to make a successful defensive action.
    *   *Data Clues:* The clearest evidence is `'maintained_safe_distance': false` in one of the loser's `'interval_analysis.retreat_analyses'`.

*   **Sub-category 3.2: Failed to "Pull Distance" vs. Lunge (Positional Error)** 
    *   *Reasoning:* The opponent committed to a lunge, but the fencer's defensive retreat was not fast or deep enough to make the attack fall short.
    *   *Data Clues:* The most specific evidence is finding a `'launch_responses'` list in a retreat analysis where an item has `'pulled_distance': false`.

*   **Sub-category 3.3: Missed Counter-Attack Opportunity (Tactical Error)**
    *   *Reasoning:* The fencer successfully created a defensive opening (e.g., the opponent paused in range) but hesitated and failed to capitalize with a counter-attack.
    *   *Data Clues:* Look for `'opportunities_missed'` being greater than 0 in the loser's `'interval_analysis.summary.defense'`.

*   **Sub-category 3.4: General/Unclassified**
    *   *Reasoning:* The loss was due to a complex interaction of timing and a high-quality attack by the opponent where no specific defensive error was obvious.
    *   *Data Clues:* This is the default choice if no other defensive rule clearly applies, suggesting the opponent's offense was simply superior.

---

### **Your Task**

Analyze the following JSON data. Identify the loser, confirm their category is "defense," and select the most fitting sub-category from the system above. Examine all candidate reasons one by one, explain their fit level, then output the best selection.

**DATA:**
```json
{JSON_DATA}
```

**OUTPUT FORMAT:**
Your response MUST be a single, valid JSON object and nothing else. Use the following structure:
```json
{
  "loss_category": "Defense",
  "loss_sub_category": "The name of the sub-category you selected",
  "brief_reasoning": "Explain main cause in English, mention why other options do not hold, and cite key data points."
}
```'''
    }
    
    return prompts


def get_win_analysis_prompts() -> Dict[str, str]:
    """Return prompts for determining winning reasons by category."""

    prompts = {
        'in_box': '''You are an elite fencing analyst. Determine the **primary winning reason** for the following touch where the fencer won in the **In-Box** phase. Follow this classification strictly:

1. "Superior Reaction at Start" (Timing Superiority)
   - Reasoning: Won the race from "Allez"; immediate advantage.
   - Data: Winner first step reaction time is faster than loser.
2. "Overpowering at Start" (Physical Dominance)
   - Reasoning: Both reacted equally fast, but winner's first-step velocity/acceleration far exceeds the loser.
   - Data: winner's velocity & acceleration are higher.
3. "Exploited Hesitation" (Tactical Acuity)
   - Reasoning: Opponent hesitated/paused; winner attacked into the vacuum.
   - Data: Loser shows pause/retreat interval.

TASK: Study the JSON, compare left/right metrics, and output the BEST fitting reason. Before making final determination, compare all candidate reasons one by one, state whether they fit the data, then select the most appropriate one.

RESPONSE (English JSON only):
{{
  "win_category": "In-Box",
  "win_sub_category": "Superior Reaction at Start | Overpowering at Start | Exploited Hesitation",
  "brief_reasoning": "Summarize in English why this reason fits best, explain why other options do not hold, cite key data.",
  "data_evidence": ["List at least two key data points, e.g.: left.init_time=0.08s vs right=0.16s", "winner.acceleration=3.2 vs loser=1.1"],
  "supporting_actions": ["Describe action rhythm/footwork observations"]
}}''',

        'attack': '''You are an elite fencing analyst. The fencer won as the attacker. Identify the primary winning reason:

1. "Superior Positioning - Optimal Distance" (Priority)
   - Reasoning: Perfect distance control; defender had no time.
   - Data: Winner's attacking interval has good_attack_distance=true; defender shows maintained_safe_distance=false and pulled_distance=false.
2. "Superior Tactics - Rhythm Break"
   - Reasoning: Winner manipulated tempo (variable/broken) to open the line.
   - Data: Winner advance tempo_type=variable_tempo/broken_tempo; 
3. "Superior Power - Overwhelming Attack"
   - Reasoning: Pure speed/power overload; defender could not cope.
   - Data: Winner attacking_velocity/acceleration high; defender retreat velocity inadequate.

Return English JSON. Before making final determination, check all candidate reasons one by one, state whether they fit, then make selection:
{{
  "win_category": "Attack",
  "win_sub_category": "Superior Positioning - Optimal Distance | Superior Tactics - Rhythm Break | Superior Power - Overwhelming Attack",
  "brief_reasoning": "Briefly explain why this reason fits best, cite data",
  "data_evidence": ["List at least two key data points"],
  "supporting_actions": ["Describe key actions/rhythm changes"]
}}''',

        'defense': '''You are an elite fencing analyst. The defender scored. Classify the winning reason:

1. "Capitalized on Attacker Positional Error" (Priority)
   - Reasoning: Attacker launched from bad distance; defender punished.
   - Data: Loser attack interval good_attack_distance=false.
2. "Capitalized on Attacker Rhythmic Error" (Inferred Parry-Riposte)
   - Reasoning: Defender reversed a predictable one-rhythm attack.
   - Data: Loser advance tempo steady_tempo; defender retreats then immediately advances/launches with high acceleration at attack climax.
3. "Capitalized on Attacker Power Failure" (Inferred Counter-Attack)
   - Reasoning: Attacker lacked speed/power; defender seized initiative with faster action.
   - Data: Loser attacking_velocity/acceleration very low; winner's corresponding metrics much higher.

Return English JSON. Before making final determination, check all candidate reasons one by one, state whether they fit, then make selection:
{{
  "win_category": "Defense",
  "win_sub_category": "Capitalized on Attacker Positional Error | Capitalized on Attacker Rhythmic Error | Capitalized on Attacker Power Failure",
  "brief_reasoning": "Cite key data to provide core argument",
  "data_evidence": ["List at least two key data points"],
  "supporting_actions": ["Describe defensive actions/counter rhythm"]
}}'''
    }

    return prompts

def _ensure_english_suffix(text: str) -> str:
    """Best-effort ensure English output by appending instruction if missing in prompt."""
    try:
        marker = "\n\nPlease respond in English, point by point, concise and powerful."
        return text + marker if 'Please respond in English' not in text else text
    except Exception:
        return text


def _normalize_category_performance(result: Dict, category: str) -> Dict:
    """Ensure required keys exist and are properly structured for enhanced analysis."""
    safe = result.copy() if isinstance(result, dict) else {}
    category_en = {'in_box': 'Inbox', 'attack': 'Attack', 'defense': 'Defense'}.get(category, category)

    defaults = {
        'performance_summary': f'This category ({category_en}) has no valid summary yet',
        'technical_analysis': 'Technical analysis not available',
        'tactical_analysis': 'Tactical analysis not available',
        'recurring_problems': [
            {
                'problem': f'{category_en} basic skills need improvement',
                'frequency': 'Data analysis in progress',
                'impact': 'Impact assessment in progress',
                'root_cause': 'Cause analysis in progress'
            }
        ],
        'detailed_improvements': [
            {
                'category': 'Technical Improvement',
                'improvements': [
            {
                'problem': f'{category_en} skills need improvement',
                        'solutions': ['Basic technique practice', 'Specialized training', 'Observation learning'],
                        'training_drills': ['Basic training', 'Specialized drills', 'Combat simulation']
                    }
                ]
            }
        ],
        'opponent_exploitation': [
            {
                'opponent_weakness': 'Opponent weakness analysis in progress',
                'exploitation_methods': ['Analyze opponent weaknesses', 'Formulate targeted tactics', 'Exploit advantages']
            }
        ],
        'competition_strategies': [
            {
                'scenario': 'General match situation',
                'strategies': ['Stable performance', 'Basic technique application', 'Maintain focus']
            }
        ],
        'priority_training_plan': [
            {
                'timeframe': 'Short-term (1-2 weeks)',
                'focus': 'Establish basic technique',
                'specific_exercises': ['Basic drills', 'Specialized training', 'Combat simulation'],
                'success_metrics': 'Skill improvement assessment'
            }
        ],
        'overall_rating': '6',
        'key_insights': ['Data analysis in progress', 'Technical assessment in progress', 'Tactical analysis in progress']
    }
    
    # Fill missing keys with defaults
    for k, v in defaults.items():
        if k not in safe or safe[k] in (None, ''):
            safe[k] = v
    
    # Validate complex nested structures
    complex_structures = {
        'recurring_problems': ['problem', 'frequency', 'impact', 'root_cause'],
        'detailed_improvements': ['category', 'improvements'],
        'opponent_exploitation': ['opponent_weakness', 'exploitation_methods'],
        'competition_strategies': ['scenario', 'strategies'],
        'priority_training_plan': ['timeframe', 'focus', 'specific_exercises', 'success_metrics']
    }
    
    for structure_name, required_keys in complex_structures.items():
        if not isinstance(safe.get(structure_name), list):
            safe[structure_name] = defaults[structure_name]
            continue

        valid_items = []
        for item in safe[structure_name]:
            if not isinstance(item, dict):
                continue
            if not all(key in item for key in required_keys):
                continue

            if structure_name == 'detailed_improvements':
                if isinstance(item.get('improvements'), list):
                    valid_items.append(item)
                continue

            if any(key in required_keys for key in ['exploitation_methods', 'strategies', 'specific_exercises']):
                list_key = next((key for key in required_keys if key in ['exploitation_methods', 'strategies', 'specific_exercises']), None)
                if list_key and isinstance(item.get(list_key), list):
                    valid_items.append(item)
                continue

            valid_items.append(item)

        safe[structure_name] = valid_items if valid_items else defaults[structure_name]
    
    # Ensure simple lists are properly formatted
    for lk in ['key_insights']:
        if not isinstance(safe.get(lk), list):
            safe[lk] = defaults[lk]
    
    # Ensure strings are properly formatted
    for sk in ['performance_summary', 'technical_analysis', 'tactical_analysis', 'overall_rating']:
        val = safe.get(sk)
        if not isinstance(val, str):
            safe[sk] = str(val) if val is not None else defaults[sk]
    
    return safe


def _normalize_overall_performance(result: Dict) -> Dict:
    """Ensure required keys exist with enhanced structure for comprehensive analysis."""
    safe = result.copy() if isinstance(result, dict) else {}
    
    defaults = {
        'performance_profile': '',
        'detailed_strengths': [],
        'critical_weaknesses': [],
        'situational_strategies': [],
        'development_roadmap': [],
        'competition_readiness': '',
        'immediate_focus': '',
        'key_insights': [],
        'rapid_adjustments': []
    }
    
    # Fill missing keys with defaults
    for k, v in defaults.items():
        if k not in safe or safe[k] in (None, ''):
            safe[k] = v
    
    # Validate detailed_strengths structure
    if not isinstance(safe.get('detailed_strengths'), list):
        safe['detailed_strengths'] = []
    else:
        valid_strengths = []
        for strength in safe['detailed_strengths']:
            if isinstance(strength, dict) and all(key in strength for key in ['strength', 'evidence', 'enhancement_strategies']):
                if isinstance(strength['enhancement_strategies'], list):
                    valid_strengths.append(strength)
        if not valid_strengths:
            safe['detailed_strengths'] = defaults['detailed_strengths']
        else:
            safe['detailed_strengths'] = valid_strengths
    
    # Validate critical_weaknesses structure
    if not isinstance(safe.get('critical_weaknesses'), list):
        safe['critical_weaknesses'] = []
    else:
        valid_weaknesses = []
        for weakness in safe['critical_weaknesses']:
            if isinstance(weakness, dict) and all(key in weakness for key in ['weakness', 'impact', 'improvement_strategies']):
                if isinstance(weakness['improvement_strategies'], list):
                    valid_weaknesses.append(weakness)
        if not valid_weaknesses:
            safe['critical_weaknesses'] = defaults['critical_weaknesses']
        else:
            safe['critical_weaknesses'] = valid_weaknesses
    
    # Validate other structured fields
    for field_name, required_keys in [
        ('situational_strategies', ['scenario', 'tactics']),
        ('development_roadmap', ['phase', 'focus', 'methods'])
    ]:
        if not isinstance(safe.get(field_name), list):
            safe[field_name] = []
        else:
            valid_items = []
            for item in safe[field_name]:
                if isinstance(item, dict) and all(key in item for key in required_keys):
                    if 'tactics' in required_keys and isinstance(item.get('tactics'), list):
                        valid_items.append(item)
                    elif 'methods' in required_keys and isinstance(item.get('methods'), list):
                        valid_items.append(item)
            if not valid_items:
                safe[field_name] = defaults[field_name]
            else:
                safe[field_name] = valid_items
    
    # Validate strings
    for sk in ['performance_profile', 'competition_readiness', 'immediate_focus']:
        val = safe.get(sk)
        if not isinstance(val, str):
            safe[sk] = str(val) if val is not None else ''

    return safe


def get_performance_analysis_prompts() -> Dict[str, str]:
    """Return the prompts for performance analysis - ENHANCED WITH SPECIFIC ADVICE"""
    
    prompts = {
        'category_analysis': '''You are a senior fencing coach. Based on detailed loss analysis, provide comprehensive tactical and technical improvement plan for {category} category:

Performance Data:
{metrics_summary}
{match_statistics}

Self Loss Analysis:
{self_loss_analysis}

Opponent Loss Analysis (understand opponent weaknesses to formulate targeted strategies):
{opponent_loss_analysis}

Self Win Patterns (to consolidate advantages):
{self_win_analysis}

Opponent Win Highlights (understand their threat sources):
{opponent_win_analysis}

**Analysis Requirements:**
1. Focus on recurring loss patterns. If the same problem appears multiple times, it must be analyzed in depth.
2. Provide very specific recommendations, avoid generalities.
3. Formulate targeted tactics based on opponent weaknesses, explain response strategy.
4. For each problem, provide multiple specific solution options, noting training frequency/verification methods.
5. Cite at least two specific data points (e.g., good_attack_distance=2/7, counter_opportunity=0/3), indicate sample size and determine if further verification is needed.
6. Start with brief explanation of category definitions (Inbox=both start simultaneously, Attack=we initiate first, Defense=opponent attacks first we defend), avoid confusion.
7. If win cause samples are very few, provide "verify first/stress test" recommendations.

Please analyze in English and return JSON:
{{
  "performance_summary": "Based on loss analysis, precisely summarize {category} performance in 3-4 sentences in English, highlighting recurring problem patterns",
  "technical_analysis": "Detailed technical analysis: including specific footwork issues, attack distance, weapon use, body coordination, etc., combined with loss causes and frequency for precise diagnosis",
  "tactical_analysis": "Detailed tactical analysis: including timing selection, distance management, rhythm control, etc., formulating targeted tactics based on opponent weaknesses",
  "recurring_problems": [
    {{
      "problem": "Main recurring problem (e.g., counterattacked multiple times due to excessive attack distance, occurred 3 times)",
      "frequency": "Number of occurrences and specific bouts",
      "impact": "Specific impact on match results",
      "root_cause": "Root cause analysis"
    }},
    {{
      "problem": "Another recurring problem",
      "frequency": "Occurrence frequency",
      "impact": "Impact analysis",
      "root_cause": "Cause analysis"
    }}
  ],
  "detailed_improvements": [
    {{
      "category": "Technical Improvement",
      "improvements": [
        {{
          "problem": "Specific technical issue (e.g., inaccurate lunge distance control)",
          "solutions": [
            "Solution 1: Specific action adjustment (e.g., use half-step to probe distance before lunge)",
            "Solution 2: Alternative technique (e.g., use continuous small steps to close in then sudden lunge)",
            "Solution 3: Complementary technique (e.g., combine feint to mask true attack distance)"
          ],
          "training_drills": [
            "Drill 1: Distance perception practice (coach moves target, student judges optimal attack timing)",
            "Drill 2: Standard distance repetition (100 lunge practices at fixed distance)",
            "Drill 3: Practical distance application (practice distance adjustment with opponents of different heights)"
          ]
        }}
      ]
    }},
    {{
      "category": "Tactical Improvement",
      "improvements": [
        {{
          "problem": "Specific tactical issue",
          "solutions": ["Tactical solution 1", "Tactical solution 2", "Tactical solution 3"],
          "training_drills": ["Tactical drill 1", "Tactical drill 2", "Tactical drill 3"]
        }}
      ]
    }}
  ],
  "opponent_exploitation": [
    {{
      "opponent_weakness": "Specific opponent weakness (e.g., often retreats late when defending)",
      "exploitation_methods": [
        "Method 1: Specific tactic (e.g., use rapid combination to suppress their retreat)",
        "Method 2: Technical application (e.g., feint to induce early retreat then real attack)",
        "Method 3: Timing selection (e.g., launch sudden attack during their habitual pause)"
      ]
    }}
  ],
  "competition_strategies": [
    {{
      "scenario": "Match scenario (e.g., when leading in score)",
      "strategies": [
        "Strategy 1: Specific execution steps and precautions",
        "Strategy 2: Alternative plans and countermeasures",
        "Strategy 3: Risk control and adjustment methods"
      ]
    }}
  ],
  "priority_training_plan": [
    {{
      "timeframe": "Short-term (1-2 weeks)",
      "focus": "Most urgent recurring problem to resolve",
      "specific_exercises": ["Specific exercise 1", "Specific exercise 2", "Specific exercise 3"],
      "success_metrics": "Improvement evaluation criteria"
    }},
    {{
      "timeframe": "Mid-term (1-2 months)",
      "focus": "Comprehensive technical and tactical improvement",
      "specific_exercises": ["Advanced exercise 1", "Advanced exercise 2", "Advanced exercise 3"],
      "success_metrics": "Improvement evaluation criteria"
    }}
  ],
  "overall_rating": "Integer rating from 1-10",
  "key_insights": [
    "Key insight 1 based on data analysis",
    "Key insight 2 based on data analysis",
    "Key insight 3 based on data analysis"
  ]
}}''',
        'overall_analysis': '''You are an Olympic-level fencing coach and data analyst. Strictly base your evaluation on the provided data. Avoid vague statements, exaggeration, or repetitive reminders. All recommendations must be actionable. If samples are scarce, mention it only once briefly in "performance_profile".

【Basic Information】
{baseline_summary}

【Category Performance】
{category_summary}

【Technical Metrics】
{metric_summary}

【Loss Patterns】
{loss_summary}

【Win Patterns】
{win_summary}

【Style Notes】
{style_summary}

【Risk Alerts】
{risk_summary}

Please output JSON following these constraints:
{{
  "performance_profile": "2-3 sentences summarizing style and current win/loss trend, cite at least 1 metric; if samples are insufficient, mention only once here.",
  "key_insights": ["Maximum 3 items, each ≤38 words, include metrics and numerator/denominator, example: 'Attack scoring rate 40% (2/5) higher than defense 0/3'"],
  "detailed_strengths": [
    {{
      "strength": "Advantage point (≤18 words, e.g., second intention connection)",
      "evidence": "Cite metrics or specific bouts to explain why it's effective",
      "enhancement_strategies": ["1-2 consolidation plans, specify training frequency or verification threshold"]
    }}
  ],
  "critical_weaknesses": [
    {{
      "weakness": "Main shortcoming (≤18 words)",
      "impact": "Explain how the shortcoming leads to losses, cite relevant data",
      "improvement_strategies": ["1-2 training/tactical measures, including frequency or achievement standards"]
    }}
  ],
  "situational_strategies": [
    {{
      "scenario": "High-value situation (e.g., trailing by two points)",
      "tactics": ["2-3 action steps, highlight rhythm/distance/decision points and cite data"]
    }},
    {{
      "scenario": "Another situation (e.g., leading to maintain win or opponent waiting for counter)",
      "tactics": ["2-3 specific execution points, cite relevant metrics"]
    }}
  ],
  "development_roadmap": [
    {{
      "phase": "Short-term (1-2 weeks)",
      "focus": "Primary issue to verify or strengthen",
      "methods": ["2 quantifiable exercises (including frequency/sample targets)"]
    }},
    {{
      "phase": "Mid-term (1-2 months)",
      "focus": "Establish stable scoring routine or address key shortcoming",
      "methods": ["2 combat or specialized training items, specify monitoring metrics"]
    }},
    {{
      "phase": "Long-term (3+ months)",
      "focus": "Tactical system or psychological adjustment goals",
      "methods": ["2 continuous tracking plans or data metrics"]
    }}
  ],
  "competition_readiness": "1 sentence assessing suitability for high-intensity competition, provide quantitative basis or supplementary testing goals.",
  "immediate_focus": "≤20 words, clarify core task for next training session."
}}

Other requirements:
- Cite at least 3 different quantitative metrics, specify denominator or frequency.
- Prohibit vague statements like "strengthen conditioning" or "improve level".
- Return empty string or empty array for fields lacking data, do not write placeholder words.
- Maintain English output.
'''
    }
    
    return prompts

# Removed complex metrics function to simplify data requirements

def _build_detailed_loss_summary(loss_analysis: Dict, fencer_side: str, category: str = None) -> str:
    """Build detailed loss summary for specific fencer and optionally specific category"""

    if not loss_analysis or f'{fencer_side}_fencer' not in loss_analysis:
        return "No loss analysis data"

    fencer_losses = loss_analysis[f'{fencer_side}_fencer']
    summary_parts = []

    # If category specified, only analyze that category
    categories_to_analyze = [category] if category else ['in_box', 'attack', 'defense']

    for cat in categories_to_analyze:
        if cat not in fencer_losses:
            continue

        category_losses = fencer_losses[cat]
        if not category_losses:
            continue

        category_name = {'in_box': 'In-Box Phase', 'attack': 'Attack Phase', 'defense': 'Defense Phase'}[cat]

        # Build detailed breakdown for each loss type, prioritizing frequent issues
        sorted_losses = sorted(category_losses.items(), key=lambda x: x[1].get('count', 0), reverse=True)
        loss_details = []

        for sub_cat, data in sorted_losses:
            count = data.get('count', 0)
            reasoning = data.get('reasoning', 'No specific reason')
            reason_display = LOSS_REASON_TRANSLATIONS.get(sub_cat, sub_cat)

            # Get specific examples from touches
            example_touches = data.get('touches', [])[:3]  # Get up to 3 examples for frequent issues
            examples = []
            for touch in example_touches:
                touch_reasoning = touch.get('reasoning', '')
                if touch_reasoning:
                    examples.append(f"Round {touch.get('touch_index', 0)+1}: {touch_reasoning}")

            # Highlight recurring problems (2+ occurrences)
            frequency_indicator = ""
            if count >= 3:
                frequency_indicator = " ⚠️ High frequency issue"
            elif count >= 2:
                frequency_indicator = " ⚠️ Recurring"

            detail_text = f"  - {reason_display}: {count} errors{frequency_indicator}"
            if examples:
                detail_text += f"\n    Specific manifestations: {'; '.join(examples)}"
            if count >= 2:
                detail_text += f"\n    ⚠️ This issue recurred {count} times, needs focused attention and improvement"

            loss_details.append(detail_text)

        if loss_details:
            summary_parts.append(f"{category_name} Loss Analysis:\n" + "\n".join(loss_details))

    return "\n\n".join(summary_parts) if summary_parts else "No significant loss patterns"

def _build_opponent_loss_summary(loss_analysis: Dict, opponent_side: str, category: str = None) -> str:
    """Build opponent's loss patterns to understand their weaknesses"""

    if not loss_analysis or f'{opponent_side}_fencer' not in loss_analysis:
        return "No opponent loss analysis data"

    opponent_losses = loss_analysis[f'{opponent_side}_fencer']
    summary_parts = []

    # If category specified, only analyze that category
    categories_to_analyze = [category] if category else ['in_box', 'attack', 'defense']

    for cat in categories_to_analyze:
        if cat not in opponent_losses:
            continue

        category_losses = opponent_losses[cat]
        if not category_losses:
            continue

        category_name = {'in_box': 'In-Box Phase', 'attack': 'Attack Phase', 'defense': 'Defense Phase'}[cat]

        # Identify opponent's main weaknesses
        sorted_losses = sorted(category_losses.items(), key=lambda x: x[1].get('count', 0), reverse=True)
        top_weaknesses = sorted_losses[:3]  # Top 3 weaknesses

        weakness_details = []
        for sub_cat, data in top_weaknesses:
            count = data.get('count', 0)
            reasoning = data.get('reasoning', '')
            reason_display = LOSS_REASON_TRANSLATIONS.get(sub_cat, sub_cat)
            weakness_details.append(f"  - {reason_display}: {count} times ({reasoning})")

        if weakness_details:
            summary_parts.append(f"Opponent {category_name} Main Weaknesses:\n" + "\n".join(weakness_details))

    return "\n\n".join(summary_parts) if summary_parts else "Opponent has no obvious weakness patterns"


def _build_detailed_win_summary(win_analysis: Dict, fencer_side: str, category: str = None) -> str:
    if not win_analysis or f'{fencer_side}_fencer' not in win_analysis:
        return "No winning data"

    fencer_wins = win_analysis[f'{fencer_side}_fencer']
    summary_parts = []
    categories_to_analyze = [category] if category else ['in_box', 'attack', 'defense']

    for cat in categories_to_analyze:
        if cat not in fencer_wins:
            continue
        category_wins = fencer_wins[cat]
        if not category_wins:
            continue

        category_name = {'in_box': 'In-Box Phase', 'attack': 'Active Attack', 'defense': 'Defensive Counter'}.get(cat, cat)
        parts = [f"[{category_name}]"]

        for reason, data in sorted(category_wins.items(), key=lambda x: x[1].get('count', 0), reverse=True):
            count = data.get('count', 0)
            reason_cn = WIN_REASON_TRANSLATIONS.get(reason, reason)
            examples = _describe_examples([touch.get('touch_index', 0) + 1 for touch in data.get('touches', []) if isinstance(touch.get('touch_index'), int)])
            reasoning_samples = data.get('reasoning_samples', [])
            main_reasoning = '; '.join(reasoning_samples[:2]) if reasoning_samples else ''
            parts.append(f"- {reason_cn}: {count} times{examples}{(' — ' + main_reasoning) if main_reasoning else ''}")

        summary_parts.append('\n'.join(parts))

    return '\n\n'.join(summary_parts) if summary_parts else "No winning data"


def _build_opponent_win_summary(win_analysis: Dict, opponent_side: str, category: str = None) -> str:
    if not win_analysis or f'{opponent_side}_fencer' not in win_analysis:
        return "No opponent winning analysis data"

    opponent_wins = win_analysis[f'{opponent_side}_fencer']
    summary_parts = []
    categories_to_analyze = [category] if category else ['in_box', 'attack', 'defense']

    for cat in categories_to_analyze:
        if cat not in opponent_wins:
            continue
        category_wins = opponent_wins[cat]
        if not category_wins:
            continue

        category_name = {'in_box': 'In-Box Phase', 'attack': 'Active Attack', 'defense': 'Defensive Counter'}.get(cat, cat)
        parts = [f"[{category_name}]"]

        for reason, data in sorted(category_wins.items(), key=lambda x: x[1].get('count', 0), reverse=True):
            count = data.get('count', 0)
            reason_cn = WIN_REASON_TRANSLATIONS.get(reason, reason)
            examples = _describe_examples([touch.get('touch_index', 0) + 1 for touch in data.get('touches', []) if isinstance(touch.get('touch_index'), int)])
            parts.append(f"- {reason_cn}: {count} times{examples}")

        summary_parts.append('\n'.join(parts))

    return '\n\n'.join(summary_parts) if summary_parts else "Opponent has no outstanding winning reasons yet"

# Removed complex technical characteristics function to simplify data requirements

def analyze_category_performance(match_data: List[Dict], category: str, fencer_side: str, upload_id: int, user_id: int, loss_analysis: Dict = None, win_analysis: Dict = None) -> Dict:
    """Generate AI-powered performance analysis for a specific category and fencer with detailed context"""
    
    prompts = get_performance_analysis_prompts()
    prompt_template = prompts['category_analysis']
    
    try:
        # Filter matches for this category and fencer
        category_matches = []
        for touch in match_data:
            fencer_category = touch.get(f'{fencer_side}_fencer_category', touch.get('bout_type', 'in_box'))
            if fencer_category == category:
                category_matches.append(touch)
        
        if not category_matches:
            return {
                'performance_summary': f'This fencer has no match data in {category} category, cannot perform detailed analysis',
                'technical_analysis': 'Insufficient data, cannot perform technical analysis',
                'tactical_analysis': 'Insufficient data, cannot perform tactical analysis',
                'recurring_problems': [
                    {
                        'problem': 'Lack of match data for this category',
                        'frequency': 'No data',
                        'impact': 'Cannot assess performance',
                        'root_cause': 'Insufficient match experience'
                    }
                ],
                'detailed_improvements': [
                    {
                        'category': 'Foundation Building',
                        'improvements': [
                            {
                                'problem': 'Lack of actual combat experience in this category',
                                'solutions': ['Increase match practice for this category', 'Focus training on this category skills', 'Watch relevant match videos to learn'],
                                'training_drills': ['Basic technical exercises', 'Specialized skill training', 'Actual combat simulation']
                            }
                        ]
                    }
                ],
                'opponent_exploitation': [
                    {
                        'opponent_weakness': 'Need more match data to analyze opponent',
                        'exploitation_methods': ['Accumulate actual combat experience', 'Observe opponent patterns', 'Learn response strategies']
                    }
                ],
                'competition_strategies': [
                    {
                        'scenario': 'First time competing',
                        'strategies': ['Accumulate match experience', 'Apply basic techniques', 'Stay focused']
                    }
                ],
                'priority_training_plan': [
                    {
                        'timeframe': 'Short-term (1-2 weeks)',
                        'focus': 'Establish basic skills',
                        'specific_exercises': ['Basic technical exercises', 'Specialized skill training', 'Actual combat simulation'],
                        'success_metrics': 'Skill mastery level'
                    }
                ],
                'overall_rating': '5',
                'key_insights': ['Need to accumulate more match data', 'Focus on developing basic skills', 'Increase actual combat practice']
            }
        
        # Calculate detailed performance metrics
        wins = sum(1 for touch in category_matches if touch.get('winner') == fencer_side)
        total = len(category_matches)
        win_rate = (wins / total * 100) if total > 0 else 0
        
        # Build enhanced metrics summary
        metrics_summary = f"Win rate: {win_rate:.1f}% ({wins} wins/{total} rounds)"

        # Build detailed match statistics with outcomes
        match_stats = []
        for i, touch in enumerate(category_matches):
            winner = touch.get('winner', 'unknown')
            result = 'Victory' if winner == fencer_side else 'Defeat'
            match_stats.append(f"Round {i+1}: {result}")

        match_statistics = f"Total rounds for this category: {total}\n" + "\n".join(match_stats[:10])
        if len(category_matches) > 10:
            match_statistics += f"\n... Plus {len(category_matches) - 10} more rounds"

        # Build detailed loss analysis for this fencer and category
        opponent_side = 'right' if fencer_side == 'left' else 'left'
        self_loss_analysis = _build_detailed_loss_summary(loss_analysis, fencer_side, category) if loss_analysis else "No loss analysis data"
        opponent_loss_analysis = _build_opponent_loss_summary(loss_analysis, opponent_side, category) if loss_analysis else "No opponent loss analysis data"
        self_win_analysis = _build_detailed_win_summary(win_analysis, fencer_side, category) if win_analysis else "No winning data"
        opponent_win_analysis = _build_opponent_win_summary(win_analysis, opponent_side, category) if win_analysis else "No opponent winning data"

        # Prepare the prompt
        category_display = {
            'in_box': 'In-Box Phase',
            'attack': 'Attack Phase',
            'defense': 'Defense Phase'
        }.get(category, category)
        
        full_prompt = prompt_template.format(
            category=category_display,
            metrics_summary=metrics_summary,
            match_statistics=match_statistics,
            self_loss_analysis=self_loss_analysis,
            opponent_loss_analysis=opponent_loss_analysis,
            self_win_analysis=self_win_analysis,
            opponent_win_analysis=opponent_win_analysis
        )
        
        # Call Gemini API
        logging.info(f"Analyzing {category} performance for {fencer_side} fencer (upload {upload_id})")
        analysis_result = call_gemini_api(full_prompt, {
            'category': category,
            'fencer': fencer_side,
            'matches': len(category_matches),
            'performance': {
                'wins': wins,
                'total': total,
                'win_rate': win_rate
            }
        })
        # Normalize to ensure keys exist and are Chinese
        return _normalize_category_performance(analysis_result, category)
        
    except Exception as e:
        logging.error(f"Error analyzing {category} performance for {fencer_side}: {e}")
        error_result = {
            'performance_summary': f'Error occurred while analyzing {category} performance',
            'technical_analysis': 'Technical analysis could not be completed due to error',
            'tactical_analysis': 'Tactical analysis could not be completed due to error',
            'specific_improvements': [
                {
                    'problem': 'Technical error occurred during analysis process',
                    'immediate_solutions': ['Review basic ' + category + ' skills', 'Manually check data', 'Re-run analysis'],
                    'training_methods': ['Basic technical exercises', 'Specialized training', 'Actual combat simulation']
                }
            ],
            'exploit_opponent_weaknesses': ['Need to re-analyze data'],
            'competition_tactics': ['Apply basic techniques', 'Stay focused', 'Stable performance'],
            'overall_rating': '5',
            'priority_focus': 'Technical foundation'
        }
        return _normalize_category_performance(error_result, category)

def analyze_overall_performance(
    match_data: List[Dict],
    fencer_side: str,
    performance_metrics: Dict,
    upload_id: int,
    user_id: int,
    loss_analysis: Dict = None,
    win_analysis: Dict = None,
    reason_briefs: Optional[Dict] = None
) -> Dict:
    """Generate overall analysis using Gemini with contextual prompts; fall back to deterministic summary."""

    total_touches = len(match_data)
    wins = sum(1 for touch in match_data if touch.get('winner') == fencer_side)
    losses = max(total_touches - wins, 0)

    category_stats = {'in_box': {'total': 0, 'wins': 0}, 'attack': {'total': 0, 'wins': 0}, 'defense': {'total': 0, 'wins': 0}}
    for touch in match_data:
        fencer_category = touch.get(f'{fencer_side}_fencer_category', touch.get('bout_type', 'in_box'))
        if fencer_category in category_stats:
            category_stats[fencer_category]['total'] += 1
            if touch.get('winner') == fencer_side:
                category_stats[fencer_category]['wins'] += 1

    category_results = _collect_category_results(category_stats)
    loss_patterns = _extract_top_loss_patterns(loss_analysis, fencer_side, max_entries=5)
    win_patterns = _extract_top_win_patterns(win_analysis, fencer_side, max_entries=5)

    loss_brief_lines: List[str] = []
    win_brief_lines: List[str] = []
    if reason_briefs and fencer_side in reason_briefs:
        side_briefs = reason_briefs[fencer_side]
        loss_map = side_briefs.get('loss') if isinstance(side_briefs, dict) else None
        win_map = side_briefs.get('win') if isinstance(side_briefs, dict) else None

        if isinstance(loss_map, dict):
            for category, bullets in loss_map.items():
                label = CATEGORY_LABELS.get(category, category)
                for bullet in bullets or []:
                    if bullet:
                        loss_brief_lines.append(f"[{label}] {bullet}")

        if isinstance(win_map, dict):
            for category, bullets in win_map.items():
                label = CATEGORY_LABELS.get(category, category)
                for bullet in bullets or []:
                    if bullet:
                        win_brief_lines.append(f"[{label}] {bullet}")

    loss_brief_lines = loss_brief_lines[:6]
    win_brief_lines = win_brief_lines[:6]

    fallback_summary = _build_overall_performance_summary(
        fencer_side=fencer_side,
        total_touches=total_touches,
        wins=wins,
        category_stats=category_stats,
        performance_metrics=performance_metrics,
        loss_analysis=loss_analysis,
        win_analysis=win_analysis,
        match_data=match_data
    )

    try:
        context_sections = _build_overall_prompt_context(
            fencer_side=fencer_side,
            total_touches=total_touches,
            wins=wins,
            losses=losses,
            category_results=category_results,
            metrics=performance_metrics,
            loss_patterns=loss_patterns,
            win_patterns=win_patterns,
            loss_brief_lines=loss_brief_lines,
            win_brief_lines=win_brief_lines
        )

        prompts = get_performance_analysis_prompts()
        prompt_template = prompts['overall_analysis']
        full_prompt = prompt_template.format(
            baseline_summary=context_sections['baseline_summary'],
            category_summary=context_sections['category_summary'],
            metric_summary=context_sections['metric_summary'],
            loss_summary=context_sections['loss_summary'],
            win_summary=context_sections['win_summary'],
            style_summary=context_sections['style_summary'],
            risk_summary=context_sections['risk_summary']
        )

        logging.info(f"Analyzing overall performance for {fencer_side} fencer (upload {upload_id}) via Gemini")
        analysis_payload = {
            'fencer': fencer_side,
            'total_touches': total_touches,
            'wins': wins,
            'losses': losses,
            'category_results': category_results,
            'metrics': performance_metrics,
            'loss_patterns': loss_patterns,
            'win_patterns': win_patterns
        }

        analysis_result = call_gemini_api(full_prompt, analysis_payload)
        if analysis_result.get('__parse_error__'):
            raise ValueError(analysis_result.get('error_detail', 'Gemini parse error'))
        if not isinstance(analysis_result, dict) or 'performance_profile' not in analysis_result:
            raise ValueError('Gemini overall analysis returned invalid structure')

        normalized = _normalize_overall_performance(analysis_result)
        normalized = _apply_overall_post_processing(normalized, category_results, total_touches, loss_patterns, win_patterns, performance_metrics)
        return normalized

    except Exception as e:
        logging.warning(f"Falling back to deterministic overall summary for {fencer_side}: {e}")
        fallback_summary = _apply_overall_post_processing(fallback_summary, category_results, total_touches, loss_patterns, win_patterns, performance_metrics)
        return _normalize_overall_performance(fallback_summary)

def analyze_touch_outcomes(match_data: List[Dict], upload_id: int, user_id: int) -> Dict[str, Dict]:
    """Analyze loss and win reasons for every touch grouped by fencer/category."""

    loss_prompts = get_loss_analysis_prompts()
    win_prompts = get_win_analysis_prompts()

    def _init_outcome_container():
        return {
            'left_fencer': {'in_box': [], 'attack': [], 'defense': []},
            'right_fencer': {'in_box': [], 'attack': [], 'defense': []}
        }

    loss_records = _init_outcome_container()
    win_records = _init_outcome_container()

    logging.info(f"Starting outcome analysis for upload {upload_id}, processing {len(match_data)} touches")

    for idx, touch_data in enumerate(match_data):
        try:
            winner = touch_data.get('winner')
            if winner not in ['left', 'right']:
                continue

            loser = 'right' if winner == 'left' else 'left'

            # Use deterministic per-fencer category resolution
            winner_category = _determine_fencer_category(touch_data, winner)
            loser_category = _determine_fencer_category(touch_data, loser)

            # Analyze losing reason
            if loser_category in loss_records[f'{loser}_fencer']:
                prompt = loss_prompts.get(loser_category)
                if prompt:
                    logging.info(f"Analyzing loss for touch {idx + 1}: {loser} fencer, category {loser_category}")
                    loss_payload = _prune_match_analysis(touch_data)
                    # Attach winner/loser metadata and categories for clarity
                    loss_payload.update({
                        'winner': winner,
                        'loser': loser,
                        'left_fencer_category': _determine_fencer_category(touch_data, 'left'),
                        'right_fencer_category': _determine_fencer_category(touch_data, 'right'),
                        'left_intention': touch_data.get('left_intention'),
                        'right_intention': touch_data.get('right_intention')
                    })
                    loss_result = call_gemini_api(prompt, loss_payload)
                    if loss_result.get('__parse_error__'):
                        logging.warning(f"Loss analysis parse error for touch {idx + 1}, skipping")
                    else:
                        loss_result['touch_index'] = idx
                        loss_result['touch_filename'] = touch_data.get('filename', f'touch_{idx + 1}')
                        loss_result['video_path'] = touch_data.get('video_path', '')
                        loss_records[f'{loser}_fencer'][loser_category].append(loss_result)
                    _sleep_with_jitter(GEMINI_TOUCH_DELAY_S)

            # Analyze winning reason
            if winner_category in win_records[f'{winner}_fencer']:
                if winner_category == 'in_box':
                    logging.info(
                        f"Classifying in-box win heuristically for touch {idx + 1}: {winner} fencer"
                    )
                    win_result = _build_inbox_win_reason(touch_data, winner)
                    win_result['touch_index'] = idx
                    win_result['touch_filename'] = touch_data.get('filename', f'touch_{idx + 1}')
                    win_result['video_path'] = touch_data.get('video_path', '')
                    win_result['source'] = 'deterministic'
                    win_records[f'{winner}_fencer'][winner_category].append(win_result)
                else:
                    prompt = win_prompts.get(winner_category)
                    if prompt:
                        logging.info(
                            f"Analyzing win for touch {idx + 1}: {winner} fencer, category {winner_category}"
                        )
                        win_payload = _prune_match_analysis(touch_data)
                        win_payload.update({
                            'winner': winner,
                            'loser': loser,
                            'left_fencer_category': _determine_fencer_category(touch_data, 'left'),
                            'right_fencer_category': _determine_fencer_category(touch_data, 'right'),
                            'left_intention': touch_data.get('left_intention'),
                            'right_intention': touch_data.get('right_intention')
                        })
                        win_result = call_gemini_api(prompt, win_payload)
                        if win_result.get('__parse_error__'):
                            logging.warning(f"Win analysis parse error for touch {idx + 1}, skipping")
                        else:
                            win_result['touch_index'] = idx
                            win_result['touch_filename'] = touch_data.get('filename', f'touch_{idx + 1}')
                            win_result['video_path'] = touch_data.get('video_path', '')
                            win_records[f'{winner}_fencer'][winner_category].append(win_result)
                        _sleep_with_jitter(GEMINI_TOUCH_DELAY_S)

        except Exception as e:
            logging.error(f"Error analyzing touch {idx}: {e}")
            continue

    def _group_outcomes(records: Dict[str, Dict[str, List[Dict]]], reason_key: str) -> Dict[str, Dict[str, Dict]]:
        grouped: Dict[str, Dict[str, Dict]] = {}
        for fencer_side, categories in records.items():
            grouped[fencer_side] = {}
            for category, outcomes in categories.items():
                reason_map: Dict[str, Dict] = {}
                for outcome in outcomes:
                    raw_sub_reason = outcome.get(reason_key, 'General/Unclassified')
                    # Normalize to canonical label to avoid duplicates like 'Sub-category 1.2: X' vs '1.2: X'
                    sub_reason = _canonical_reason_label(raw_sub_reason)
                    data_entry = reason_map.setdefault(sub_reason, {
                        'count': 0,
                        'touches': [],
                        'reasoning_samples': [],
                        'supporting_points': []
                    })
                    data_entry['count'] += 1
                    reasoning = outcome.get('brief_reasoning') or ''
                    if reasoning and reasoning not in data_entry['reasoning_samples']:
                        data_entry['reasoning_samples'].append(reasoning)
                    if reasoning and not data_entry.get('reasoning'):
                        data_entry['reasoning'] = reasoning
                    support = outcome.get('supporting_actions') or outcome.get('supporting_points') or []
                    if isinstance(support, list):
                        for item in support:
                            if item and item not in data_entry['supporting_points']:
                                data_entry['supporting_points'].append(item)
                    touch_entry = {
                        'touch_index': outcome.get('touch_index', 0),
                        'filename': outcome.get('touch_filename', ''),
                        'video_path': outcome.get('video_path', ''),
                        'reasoning': reasoning,
                        'data_evidence': outcome.get('data_evidence', [])
                    }
                    data_entry['touches'].append(touch_entry)
                grouped[fencer_side][category] = reason_map
        return grouped

    grouped_loss = _group_outcomes(loss_records, 'loss_sub_category')
    grouped_win = _group_outcomes(win_records, 'win_sub_category')

    logging.info(f"Completed outcome analysis for upload {upload_id}")
    return {
        'loss_grouped': grouped_loss,
        'win_grouped': grouped_win,
        'raw_loss': loss_records,
        'raw_win': win_records
    }


def _mean_or_none(values: List[float]) -> Optional[float]:
    cleaned = [v for v in values if v is not None]
    return round(mean(cleaned), 4) if cleaned else None


def _truncate_text(text: str, limit: int = 900) -> str:
    if not isinstance(text, str):
        return ''
    return text if len(text) <= limit else text[:limit].rstrip() + '...'


def _condense_sentence(text: str, max_sentences: int = 2, max_length: int = 200) -> str:
    """Return text limited to a small number of sentences and characters."""
    if not isinstance(text, str):
        return ''
    cleaned = re.sub(r'\s+', ' ', text).strip()
    if not cleaned:
        return ''
    sentences = re.split(r'(?<=[.!?])\s+', cleaned)
    condensed = ' '.join(sentences[:max_sentences])
    if len(condensed) > max_length:
        condensed = condensed[:max_length].rstrip() + '…'
    return condensed


def _condense_list(entries: List[Any], max_items: int = 3, item_length: int = 120) -> List[str]:
    """Trim list-like Gemini outputs to a concise list of short strings."""
    if not isinstance(entries, list):
        return []
    condensed: List[str] = []
    for entry in entries:
        if entry is None or entry == '':
            continue
        text = entry if isinstance(entry, str) else json.dumps(entry, ensure_ascii=False)
        text = re.sub(r'\s+', ' ', text).strip()
        if not text:
            continue
        if len(text) > item_length:
            text = text[:item_length].rstrip() + '…'
        condensed.append(text)
        if len(condensed) >= max_items:
            break
    return condensed


def _strip_category_prefix(text: str) -> str:
    if not isinstance(text, str):
        return ''
    cleaned = text.strip()
    if cleaned.startswith('['):
        closing = cleaned.find(']')
        if closing != -1:
            cleaned = cleaned[closing + 1 :].strip()
    return cleaned


def _collect_touch_context(match_entry: Dict, fencer_side: str) -> Dict:
    """Extract key numeric signals from a match entry for Gemini prompts."""
    opponent_side = 'right' if fencer_side == 'left' else 'left'
    self_data = match_entry.get(f'{fencer_side}_data', {}) or {}
    opp_data = match_entry.get(f'{opponent_side}_data', {}) or {}

    self_step = self_data.get('first_step', {}) or {}
    opp_step = opp_data.get('first_step', {}) or {}

    self_init = _safe_float(self_step.get('init_time'))
    opp_init = _safe_float(opp_step.get('init_time'))
    reaction_advantage = None
    if self_init is not None and opp_init is not None:
        reaction_advantage = opp_init - self_init  # positive => faster than opponent

    self_velocity = _safe_float(self_step.get('velocity'))
    opp_velocity = _safe_float(opp_step.get('velocity'))
    velocity_diff = None
    if self_velocity is not None and opp_velocity is not None:
        velocity_diff = self_velocity - opp_velocity

    self_acc = _safe_float(self_step.get('acceleration'))
    opp_acc = _safe_float(opp_step.get('acceleration'))
    acceleration_diff = None
    if self_acc is not None and opp_acc is not None:
        acceleration_diff = self_acc - opp_acc

    interval_analysis = self_data.get('interval_analysis', {}) or {}
    advance_analyses = interval_analysis.get('advance_analyses', []) or []
    good_attack_distances = sum(1 for adv in advance_analyses if adv.get('good_attack_distance'))
    total_attacking_adv = sum(1 for adv in advance_analyses if adv.get('attack_info', {}).get('has_attack'))

    defense_summary = interval_analysis.get('summary', {}).get('defense', {}) or {}

    first_pause_self = _first_pause_window(self_data)
    first_pause_opp = _first_pause_window(opp_data)

    fps = _safe_float(match_entry.get('fps')) or 30.0
    frame_range = match_entry.get('frame_range') or [0, 0]
    try:
        duration = (frame_range[1] - frame_range[0]) / fps if fps and frame_range else None
    except Exception:
        duration = None

    return {
        'touch_index': match_entry.get('touch_index'),
        'match_idx': match_entry.get('match_idx'),
        'winner': match_entry.get('winner'),
        'category': match_entry.get(f'{fencer_side}_fencer_category') or match_entry.get('bout_type'),
        'first_step': {
            'self': {
                'init_time': self_init,
                'velocity': self_velocity,
                'acceleration': self_acc
            },
            'opponent': {
                'init_time': opp_init,
                'velocity': opp_velocity,
                'acceleration': opp_acc
            },
            'reaction_advantage': reaction_advantage,
            'velocity_diff': velocity_diff,
            'acceleration_diff': acceleration_diff
        },
        'tempo': {
            'self_pause_ratio': _safe_float(self_data.get('pause_ratio')),
            'opponent_pause_ratio': _safe_float(opp_data.get('pause_ratio')),
            'self_first_pause': first_pause_self,
            'opponent_first_pause': first_pause_opp
        },
        'attack_metrics': {
            'self_attacking_velocity': _safe_float(
                self_data.get('attacking_velocity') or self_data.get('summary_metrics', {}).get('attacking_velocity')
            ),
            'self_attacking_acceleration': _safe_float(
                self_data.get('attacking_acceleration') or self_data.get('summary_metrics', {}).get('attacking_acceleration')
            ),
            'good_attack_distance_count': good_attack_distances,
            'attack_with_distance_samples': total_attacking_adv
        },
        'defense_metrics': {
            'counter_opportunities': defense_summary.get('counter_opportunities'),
            'counters_executed': defense_summary.get('counters_executed'),
            'good_distance_management': defense_summary.get('good_distance_management')
        },
        'duration_seconds': duration,
        'gpt_analysis_excerpt': _truncate_text(match_entry.get('gpt_analysis', '')),
        'video_path': match_entry.get('video_path', ''),
        'filename': match_entry.get('filename')
    }


def _build_enriched_touch_entry(
    touch: Dict,
    context: Dict,
    *,
    reason_key: str,
    reason_label: str,
    reason_type: str,
    fencer_side: str
) -> Dict:
    enriched = {
        'touch_index': touch.get('touch_index'),
        'filename': touch.get('filename'),
        'video_path': touch.get('video_path'),
        'match_idx': context.get('match_idx'),
        'reasoning': touch.get('reasoning'),
        'data_evidence': touch.get('data_evidence'),
        'first_step': context.get('first_step'),
        'tempo': context.get('tempo'),
        'gpt_analysis_excerpt': context.get('gpt_analysis_excerpt'),
        'category': context.get('category'),
        'reason_key': reason_key,
        'reason_label': reason_label,
        'reason_type': reason_type,
        'fencer_side': fencer_side
    }
    return enriched


def _prune_interval_segment(segment: Dict) -> Dict:
    if not isinstance(segment, dict):
        return {}
    pruned = {}
    for key in ['interval', 'duration_frames', 'duration_seconds', 'tempo_type', 'tempo_description', 'tempo_changes', 'has_micro_pauses', 'good_attack_distance', 'missed_opportunities', 'tactical_notes']:
        value = segment.get(key)
        if value is not None:
            pruned[key] = value

    attack_info = segment.get('attack_info')
    if isinstance(attack_info, dict):
        pruned['attack_info'] = {
            'has_attack': attack_info.get('has_attack'),
            'attack_type': attack_info.get('attack_type'),
            'num_extensions': attack_info.get('num_extensions'),
            'num_launches': attack_info.get('num_launches'),
            'characteristics': attack_info.get('characteristics')
        }

    return sanitize_data_structure(pruned)


def _prune_fencer_data(data: Dict) -> Dict:
    if not isinstance(data, dict):
        return {}

    pruned = {}
    keys_to_copy = [
        'velocity', 'acceleration', 'advance_ratio', 'pause_ratio',
        'arm_extension_freq', 'attack_success_rate', 'counter_success_rate',
        'advance_sec', 'pause_sec', 'latest_pause_retreat_end',
        'launch_promptness', 'launch_frame', 'has_launch',
        'avg_arm_extension_duration', 'attacking_score'
    ]
    for key in keys_to_copy:
        value = data.get(key)
        if value is not None:
            pruned[key] = value

    if 'first_step' in data:
        pruned['first_step'] = data.get('first_step')

    summary_metrics = data.get('summary_metrics')
    if isinstance(summary_metrics, dict):
        pruned['summary_metrics'] = summary_metrics

    interval_analysis = data.get('interval_analysis') or {}
    summary = interval_analysis.get('summary')
    if summary:
        pruned['interval_summary'] = summary

    advance_segments = interval_analysis.get('advance_analyses') or []
    if advance_segments:
        pruned['advance_analyses'] = [_prune_interval_segment(seg) for seg in advance_segments[:2]]

    retreat_segments = interval_analysis.get('retreat_analyses') or []
    if retreat_segments:
        pruned['retreat_analyses'] = [_prune_interval_segment(seg) for seg in retreat_segments[:2]]

    arm_extensions = data.get('arm_extensions_sec') or []
    if arm_extensions:
        pruned['arm_extensions_summary'] = {
            'count': len(arm_extensions),
            'samples': arm_extensions[:2]
        }

    return sanitize_data_structure(pruned)


def _prune_bout_statistics(stats: Dict) -> Dict:
    if not isinstance(stats, dict):
        return {}
    pruned = {}
    for key in ['total_duration_seconds', 'touch_count', 'score_progression']:
        value = stats.get(key)
        if value is not None:
            pruned[key] = value

    for summary_key in ['left_fencer_summary', 'right_fencer_summary']:
        summary = stats.get(summary_key)
        if summary:
            pruned[summary_key] = summary

    return sanitize_data_structure(pruned)


def _prune_match_analysis(match_entry: Dict) -> Dict:
    if not isinstance(match_entry, dict):
        return {}

    pruned = {
        'match_idx': match_entry.get('match_idx'),
        'filename': match_entry.get('filename'),
        'video_path': match_entry.get('video_path'),
        'bout_type': match_entry.get('bout_type'),
        'fps': match_entry.get('fps'),
        'frame_range': match_entry.get('frame_range'),
        'video_angle': match_entry.get('video_angle'),
        'gpt_analysis': _truncate_text(match_entry.get('gpt_analysis', ''), limit=1600)
    }

    judgement = match_entry.get('judgement')
    if judgement:
        pruned['judgement'] = judgement

    if 'left_data' in match_entry:
        pruned['left_data'] = _prune_fencer_data(match_entry.get('left_data'))
    if 'right_data' in match_entry:
        pruned['right_data'] = _prune_fencer_data(match_entry.get('right_data'))
    if 'bout_statistics' in match_entry:
        pruned['bout_statistics'] = _prune_bout_statistics(match_entry.get('bout_statistics'))

    return sanitize_data_structure(pruned)


def get_reason_synthesis_prompt(reason_type: str) -> str:
    """Prompt template for deep-dive reason synthesis."""
    if reason_type == 'loss':
        return '''You are an Olympic-level fencing tactical analyst. The following is detailed data on a fencer's losses in a specific category, including bout-by-bout statistics and coach notes. Diagnose "why the loss occurred" and "how to correct it" in a concise format suitable for quick bout-side review.

Data (JSON):
```json
{JSON_DATA}
```

Analysis Requirements (keep total response under 120 words):
1. Cite specific data (milliseconds, velocity, frames, or counts). No conditioning or training-program references.
2. Provide a single headline (≤12 words) naming the failure mechanism.
3. Give a short narrative (≤2 sentences) covering decision, rhythm, distance, and execution.
4. List at most 2 key sequences with bout number or timestamp and data evidence.
5. List at most 3 correction points phrased as immediate tactical adjustments with measurable cues.
6. Provide up to 2 rapid validation checks for the next bout (no training drills).

Please output in JSON:
{
  "analysis_headline": "≤12 word conclusion naming the failure mechanism",
  "core_narrative": "≤2 short sentences covering decision, rhythm, distance, execution with data",
  "key_sequences": ["Sequence 1 with bout mark + metric", "Sequence 2 with bout mark + metric"],
  "focus_points": ["Immediate adjustment with measurable cue", "Second adjustment"],
  "validation_checks": ["Next-bout verification", "Optional second verification"],
  "summary_bullet": "≤18 word summary statement"
}
'''

    # reason_type == 'win'
    return '''You are an Olympic-level fencing tactical analyst. The following is detailed data on a fencer's wins in a specific category, including bout-by-bout statistics and coach notes. Explain "why the win occurred" and "how to repeat it" in a concise format for quick bout-side use.

Data (JSON):
```json
{JSON_DATA}
```

Analysis Requirements (keep total response under 120 words):
1. Cite concrete data (reaction, velocity, counts). No conditioning/training advice.
2. Provide one headline (≤12 words) naming the scoring principle.
3. Give a short narrative (≤2 sentences) covering trigger signal, execution continuity, and opponent reaction.
4. List at most 2 key sequences with bout mark/timestamp and the decisive cues.
5. Provide up to 3 replication points as immediate tactical rules with data thresholds.
6. Provide up to 2 stress-test checks for the next bout (no training drills).

Please output in JSON:
{
  "analysis_headline": "≤12 word description of scoring principle",
  "core_narrative": "≤2 short sentences covering trigger, execution, opponent reaction with data",
  "key_sequences": ["Sequence 1 with bout mark + metric", "Sequence 2 with bout mark + metric"],
  "focus_points": ["Replication rule 1 with cue", "Replication rule 2"],
  "validation_checks": ["Immediate stress test", "Optional second check"],
  "summary_bullet": "≤18 word summary statement"
}
'''


def _normalize_reason_report(
    result: Dict,
    reason_key: str,
    reason_info: Dict,
    reason_label: str,
    enriched_touches: List[Dict],
    aggregated_metrics: Dict,
    reason_type: str,
    fencer_side: str,
    touch_contexts: List[Dict],
    match_snapshots: List[Dict]
) -> Dict:
    """Ensure Gemini output is well-formed; provide fallback if parsing failed."""
    fallback_headline = f"{reason_label}: Insufficient data, using raw statistics" if result.get('__parse_error__') else None

    report = {
        'reason_key': reason_key,
        'reason_label': reason_label,
        'touch_count': reason_info.get('count', len(enriched_touches)),
        'analysis_headline': '',
        'core_narrative': '',
        'key_sequences': [],
        'focus_points': [],
        'validation_checks': [],
        'summary_bullet': '',
        'touches': enriched_touches,
        'aggregated_metrics': aggregated_metrics,
        'source': 'gemini',
        'reason_type': reason_type,
        'fencer_side': fencer_side,
        'touch_contexts': touch_contexts,
        'match_snapshots': match_snapshots,
        'reason_context': sanitize_data_structure({
            'count': reason_info.get('count'),
            'reasoning_samples': reason_info.get('reasoning_samples', []),
            'supporting_points': reason_info.get('supporting_points', []),
            'source_reasoning': reason_info.get('reasoning')
        })
    }

    if result.get('__parse_error__'):
        report['analysis_headline'] = _condense_sentence(
            fallback_headline or f"{reason_label} - Auto-inferred",
            max_sentences=1,
            max_length=110
        )
        report['core_narrative'] = _condense_sentence(
            reason_info.get('reasoning', 'AI parsing failed, preserving original conclusion.'),
            max_sentences=2,
            max_length=180
        )
        support_points = reason_info.get('supporting_points', [])
        report['key_sequences'] = _condense_list(support_points, max_items=2, item_length=110)
        report['focus_points'] = _condense_list(support_points, max_items=3, item_length=110)
        report['validation_checks'] = []
        report['summary_bullet'] = _condense_sentence(
            f"{reason_label}: Occurred {report['touch_count']} times, requires manual review.",
            max_sentences=1,
            max_length=110
        )
        report['source'] = 'fallback'
        return report

    report['analysis_headline'] = _condense_sentence(
        result.get('analysis_headline') or fallback_headline or f"{reason_label} - Data interpretation",
        max_sentences=1,
        max_length=110
    )
    report['core_narrative'] = _condense_sentence(
        result.get('core_narrative') or reason_info.get('reasoning', ''),
        max_sentences=2,
        max_length=180
    )
    report['key_sequences'] = _condense_list(
        result.get('key_sequences') or reason_info.get('supporting_points', [])[:3],
        max_items=2,
        item_length=110
    )
    report['focus_points'] = _condense_list(result.get('focus_points') or [], max_items=3, item_length=110)
    report['validation_checks'] = _condense_list(result.get('validation_checks') or [], max_items=2, item_length=100)
    report['summary_bullet'] = _condense_sentence(
        result.get('summary_bullet') or f"{reason_label}: Occurred {report['touch_count']} times",
        max_sentences=1,
        max_length=110
    )
    return report


def build_reason_reports(
    match_data: List[Dict],
    grouped_analysis: Dict,
    reason_type: str,
    upload_id: int,
    user_id: int
) -> Tuple[Dict, Dict]:
    """Generate deep-dive reports for each win/loss reason."""
    match_lookup = {
        entry.get('touch_index'): entry
        for entry in match_data
        if isinstance(entry.get('touch_index'), int)
    }

    reports = {
        'left_fencer': {'in_box': [], 'attack': [], 'defense': []},
        'right_fencer': {'in_box': [], 'attack': [], 'defense': []}
    }
    summary_bullets = {
        'left_fencer': {'in_box': [], 'attack': [], 'defense': []},
        'right_fencer': {'in_box': [], 'attack': [], 'defense': []}
    }

    for fencer_key in ['left_fencer', 'right_fencer']:
        fencer_side = 'left' if fencer_key == 'left_fencer' else 'right'
        category_map = grouped_analysis.get(fencer_key, {}) or {}

        for category, reasons in category_map.items():
            category_reports: List[Dict] = []
            category_bullets: List[str] = []

            sorted_reasons = sorted(
                (reasons or {}).items(),
                key=lambda item: (item[1] or {}).get('count', 0),
                reverse=True
            )

            for reason_key, info in sorted_reasons:
                touches = (info.get('touches') or [])[:4]
                if not touches:
                    continue

                reason_label = WIN_REASON_TRANSLATIONS.get(
                    reason_key,
                    LOSS_REASON_TRANSLATIONS.get(reason_key, reason_key)
                )

                touch_contexts: List[Dict] = []
                enriched_touches: List[Dict] = []
                match_snapshots: List[Dict] = []
                gemini_touch_payloads: List[Dict] = []
                reaction_advantages: List[Optional[float]] = []
                velocity_diffs: List[Optional[float]] = []
                acceleration_diffs: List[Optional[float]] = []
                self_pause_ratios: List[Optional[float]] = []
                opp_pause_ratios: List[Optional[float]] = []

                for touch in touches:
                    touch_entry = match_lookup.get(touch.get('touch_index'))
                    if not touch_entry:
                        continue

                    raw_context = _collect_touch_context(touch_entry, fencer_side)
                    raw_context['reasoning_note'] = touch.get('reasoning')
                    raw_context['data_evidence'] = touch.get('data_evidence')
                    raw_context['reason_key'] = reason_key
                    raw_context['reason_label'] = reason_label
                    clean_context = sanitize_data_structure(raw_context)
                    touch_contexts.append(clean_context)

                    enriched_touches.append(
                        _build_enriched_touch_entry(
                            touch,
                            clean_context,
                            reason_key=reason_key,
                            reason_label=reason_label,
                            reason_type=reason_type,
                            fencer_side=fencer_side
                        )
                    )

                    match_snapshots.append(sanitize_data_structure({
                        'touch_index': touch.get('touch_index'),
                        'match_idx': clean_context.get('match_idx'),
                        'filename': touch.get('filename'),
                        'video_path': touch.get('video_path'),
                        'reason_key': reason_key,
                        'reason_label': reason_label,
                        'reason_type': reason_type,
                        'fencer_side': fencer_side,
                        'gpt_analysis_excerpt': clean_context.get('gpt_analysis_excerpt')
                    }))

                    gemini_touch_payloads.append(sanitize_data_structure({
                        'touch_index': touch.get('touch_index'),
                        'match_idx': clean_context.get('match_idx'),
                        'filename': touch.get('filename'),
                        'video_path': touch.get('video_path'),
                        'fencer_side': fencer_side,
                        'reason_key': reason_key,
                        'reason_label': reason_label,
                        'reason_type': reason_type,
                        'match_analysis': _prune_match_analysis(touch_entry),
                        'reasoning_note': touch.get('reasoning'),
                        'data_evidence': touch.get('data_evidence')
                    }))

                    metrics = clean_context.get('first_step', {}) or {}
                    reaction_advantages.append(metrics.get('reaction_advantage'))
                    velocity_diffs.append(metrics.get('velocity_diff'))
                    acceleration_diffs.append(metrics.get('acceleration_diff'))
                    tempo = clean_context.get('tempo', {}) or {}
                    self_pause_ratios.append(tempo.get('self_pause_ratio'))
                    opp_pause_ratios.append(tempo.get('opponent_pause_ratio'))

                if not touch_contexts:
                    continue

                aggregated_metrics = sanitize_data_structure({
                    'reaction_advantage_avg': _mean_or_none(reaction_advantages),
                    'velocity_diff_avg': _mean_or_none(velocity_diffs),
                    'acceleration_diff_avg': _mean_or_none(acceleration_diffs),
                    'self_pause_ratio_avg': _mean_or_none(self_pause_ratios),
                    'opponent_pause_ratio_avg': _mean_or_none(opp_pause_ratios)
                })

                reason_context_summary = sanitize_data_structure({
                    'count': info.get('count'),
                    'reasoning_samples': info.get('reasoning_samples') or [],
                    'supporting_points': info.get('supporting_points') or [],
                    'source_reasoning': info.get('reasoning')
                })

                payload = {
                    'upload_id': upload_id,
                    'user_id': user_id,
                    'fencer_side': fencer_side,
                    'fencer_label': SIDE_LABELS.get(fencer_side, fencer_side),
                    'reason_type': reason_type,
                    'category_key': category,
                    'category_label': CATEGORY_LABELS.get(category, category),
                    'reason_key': reason_key,
                    'reason_label': reason_label,
                    'touch_count': len(touch_contexts),
                    'aggregated_metrics': aggregated_metrics,
                    'touch_summaries': touch_contexts,
                    'touch_match_details': gemini_touch_payloads,
                    'reason_context': reason_context_summary
                }

                prompt = get_reason_synthesis_prompt(reason_type)
                gemini_result = call_gemini_api(prompt, payload)
                report = _normalize_reason_report(
                    gemini_result,
                    reason_key,
                    info,
                    reason_label,
                    enriched_touches,
                    aggregated_metrics,
                    reason_type,
                    fencer_side,
                    touch_contexts,
                    match_snapshots
                )

                category_reports.append(report)
                if report.get('summary_bullet'):
                    category_bullets.append(report['summary_bullet'])

            reports[fencer_key][category] = category_reports
            summary_bullets[fencer_key][category] = category_bullets

    return reports, summary_bullets

def _derive_immediate_adjustments(reason_briefs: Optional[Dict], max_items_per_category: int = 2) -> Dict[str, Dict[str, List[str]]]:
    """Pull the most urgent tactical adjustments from loss summaries, structured by category (inbox/attack/defense)."""
    structured_adjustments = {
        'in_box': {'corrections': [], 'leverage': []},
        'attack': {'corrections': [], 'leverage': []},
        'defense': {'corrections': [], 'leverage': []}
    }

    if not isinstance(reason_briefs, dict):
        return structured_adjustments

    def _append_category_items(source_map: Dict, category: str, adjustment_type: str, max_items: int) -> None:
        """Extract items for a specific category and type (corrections or leverage)."""
        if not isinstance(source_map, dict):
            return

        items_list = source_map.get(category, []) or []
        for bullet in items_list:
            text = _strip_category_prefix(bullet)
            if not text:
                continue
            lower = text.lower()
            # Skip training/drill suggestions - focus on immediate tactical adjustments
            if 'training' in lower or 'drill' in lower or 'conditioning' in lower:
                continue

            formatted = text.strip()
            if not formatted:
                continue
            if not formatted.endswith(('.', '!', '?')):
                formatted += '.'

            if formatted not in structured_adjustments[category][adjustment_type]:
                structured_adjustments[category][adjustment_type].append(formatted)
            if len(structured_adjustments[category][adjustment_type]) >= max_items:
                return

    # Process loss reasons as corrections
    loss_map = reason_briefs.get('loss') or {}
    for category in ['in_box', 'attack', 'defense']:
        _append_category_items(loss_map, category, 'corrections', max_items_per_category)

    # Process win reasons as leverage points
    win_map = reason_briefs.get('win') or {}
    for category in ['in_box', 'attack', 'defense']:
        _append_category_items(win_map, category, 'leverage', max_items_per_category)

    return structured_adjustments


def generate_video_view_data(upload_id: int, user_id: int) -> Dict:
    """Main function to generate all data for video view"""
    try:
        performance_data = calculate_performance_metrics(upload_id, user_id)
        
        # Load match data for detailed analysis
        result_dir = f"results/{user_id}/{upload_id}"
        match_analysis_dir = os.path.join(result_dir, "match_analysis")
        
        match_data = []
        for filename in os.listdir(match_analysis_dir):
            if filename.endswith('_analysis.json'):
                filepath = os.path.join(match_analysis_dir, filename)
                try:
                    with open(filepath, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                        # Inject filename and derived video path/match index for later use (loss analysis/video embedding)
                        data['filename'] = filename
                        match_idx = None
                        m = re.search(r'match_(\d+)_analysis\.json$', filename)
                        if m:
                            try:
                                match_idx = int(m.group(1))
                            except Exception:
                                match_idx = None
                        data['match_idx'] = match_idx
                        if match_idx is not None:
                            data['video_path'] = _get_display_video_relpath(user_id, upload_id, match_idx)
                        else:
                            data['video_path'] = ''
                        data['touch_index'] = len(match_data)
                        match_data.append(data)
                except Exception as e:
                    logging.error(f"Error loading {filepath}: {e}")
                    continue
        
        # Calculate detailed category analyses
        inbox_analysis = calculate_inbox_analysis(match_data)
        attack_analysis = calculate_attack_analysis(match_data)
        defense_analysis = calculate_defense_analysis(match_data)
        
        # Calculate outcome analyses using Gemini API
        outcome_analysis = analyze_touch_outcomes(match_data, upload_id, user_id)
        loss_analysis = outcome_analysis['loss_grouped']
        win_analysis = outcome_analysis['win_grouped']

        win_reason_reports, win_reason_briefs = build_reason_reports(match_data, win_analysis, 'win', upload_id, user_id)
        loss_reason_reports, loss_reason_briefs = build_reason_reports(match_data, loss_analysis, 'loss', upload_id, user_id)

        reason_briefs_map = {
            'left': {
                'win': win_reason_briefs.get('left_fencer', {}),
                'loss': loss_reason_briefs.get('left_fencer', {})
            },
            'right': {
                'win': win_reason_briefs.get('right_fencer', {}),
                'loss': loss_reason_briefs.get('right_fencer', {})
            }
        }

        immediate_adjustments = {
            'left': _derive_immediate_adjustments(reason_briefs_map.get('left')),
            'right': _derive_immediate_adjustments(reason_briefs_map.get('right'))
        }

        # Generate comprehensive performance analysis for both fencers
        logging.info(f"Generating comprehensive performance analysis for upload {upload_id}")

        # Overall performance analysis (with delays between API calls)
        left_overall_analysis = analyze_overall_performance(
            match_data,
            'left',
            performance_data['left_fencer_metrics'],
            upload_id,
            user_id,
            loss_analysis,
            win_analysis,
            reason_briefs_map
        )
        _sleep_with_jitter(GEMINI_TOUCH_DELAY_S)
        right_overall_analysis = analyze_overall_performance(
            match_data,
            'right',
            performance_data['right_fencer_metrics'],
            upload_id,
            user_id,
            loss_analysis,
            win_analysis,
            reason_briefs_map
        )
        
        # Category-specific performance analysis (with delays between API calls)
        category_performance_analysis = {'left_fencer': {}, 'right_fencer': {}}

        if isinstance(left_overall_analysis, dict):
            left_overall_analysis['rapid_adjustments'] = immediate_adjustments['left']
        if isinstance(right_overall_analysis, dict):
            right_overall_analysis['rapid_adjustments'] = immediate_adjustments['right']

        # Left fencer analysis
        category_performance_analysis['left_fencer']['in_box'] = analyze_category_performance(match_data, 'in_box', 'left', upload_id, user_id, loss_analysis, win_analysis)
        _sleep_with_jitter(GEMINI_TOUCH_DELAY_S)
        category_performance_analysis['left_fencer']['attack'] = analyze_category_performance(match_data, 'attack', 'left', upload_id, user_id, loss_analysis, win_analysis)
        _sleep_with_jitter(GEMINI_TOUCH_DELAY_S)
        category_performance_analysis['left_fencer']['defense'] = analyze_category_performance(match_data, 'defense', 'left', upload_id, user_id, loss_analysis, win_analysis)
        _sleep_with_jitter(GEMINI_TOUCH_DELAY_S)
        
        # Right fencer analysis
        category_performance_analysis['right_fencer']['in_box'] = analyze_category_performance(match_data, 'in_box', 'right', upload_id, user_id, loss_analysis, win_analysis)
        _sleep_with_jitter(GEMINI_TOUCH_DELAY_S)
        category_performance_analysis['right_fencer']['attack'] = analyze_category_performance(match_data, 'attack', 'right', upload_id, user_id, loss_analysis, win_analysis)
        _sleep_with_jitter(GEMINI_TOUCH_DELAY_S)
        category_performance_analysis['right_fencer']['defense'] = analyze_category_performance(match_data, 'defense', 'right', upload_id, user_id, loss_analysis, win_analysis)

        # Overall performance analysis
        overall_performance_analysis = {
            'left_fencer': left_overall_analysis,
            'right_fencer': right_overall_analysis
        }

        # Format for radar chart (9 metrics in specific order)
        radar_data = {
            'left_fencer': format_radar_data(performance_data['left_fencer_metrics']),
            'right_fencer': format_radar_data(performance_data['right_fencer_metrics'])
        }

        detailed_analysis = build_detailed_analysis(inbox_analysis, attack_analysis, defense_analysis)
        category_chart_images = generate_mirror_chart_images(detailed_analysis)
        touch_summary = _build_touch_summary(match_data)

        result = {
            'success': True,
            'radar_data': radar_data,
            'bout_type_stats': performance_data['bout_type_statistics'],
            'total_touches': performance_data['total_touches'],
            'inbox_analysis': inbox_analysis,
            'attack_analysis': attack_analysis,
            'defense_analysis': defense_analysis,
            'detailed_analysis': detailed_analysis,
            'category_chart_images': category_chart_images,
            'touch_summary': touch_summary,
            'loss_analysis': loss_analysis,
            'win_analysis': win_analysis,
            'loss_reason_reports': loss_reason_reports,
            'win_reason_reports': win_reason_reports,
            'reason_summary_bullets': reason_briefs_map,
            'category_performance_analysis': category_performance_analysis,
            'overall_performance_analysis': overall_performance_analysis
        }
        
        # Sanitize all data to ensure JSON serializability
        return sanitize_data_structure(result)
        
    except Exception as e:
        logging.error(f"Error generating video view data for upload {upload_id}: {e}")
        import traceback
        traceback.print_exc()
        return {
            'success': False,
            'error': str(e)
        }


def get_basic_video_data(upload_id: int, user_id: int) -> Dict:
    """Get basic video data (non-AI components) for video view"""
    try:
        performance_data = calculate_performance_metrics(upload_id, user_id)
        
        # Load match data for detailed analysis
        result_dir = f"results/{user_id}/{upload_id}"
        match_analysis_dir = os.path.join(result_dir, "match_analysis")
        
        match_data = []
        for filename in os.listdir(match_analysis_dir):
            if filename.endswith('_analysis.json'):
                filepath = os.path.join(match_analysis_dir, filename)
                try:
                    with open(filepath, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                        # Inject filename and derived video path/match index for later use
                        data['filename'] = filename
                        match_idx = None
                        m = re.search(r'match_(\d+)_analysis\.json$', filename)
                        if m:
                            try:
                                match_idx = int(m.group(1))
                            except Exception:
                                match_idx = None
                        data['match_idx'] = match_idx
                        if match_idx is not None:
                            data['video_path'] = _get_display_video_relpath(user_id, upload_id, match_idx)
                        else:
                            data['video_path'] = ''
                        data['touch_index'] = len(match_data)
                        match_data.append(data)
                except Exception as e:
                    logging.error(f"Error loading {filepath}: {e}")
                    continue
        
        # Calculate detailed category analyses (non-AI parts only)
        inbox_analysis = calculate_inbox_analysis(match_data)
        attack_analysis = calculate_attack_analysis(match_data)
        defense_analysis = calculate_defense_analysis(match_data)
        
        # Format for radar chart (9 metrics in specific order)
        radar_data = {
            'left_fencer': format_radar_data(performance_data['left_fencer_metrics']),
            'right_fencer': format_radar_data(performance_data['right_fencer_metrics'])
        }
        
        detailed_analysis = build_detailed_analysis(inbox_analysis, attack_analysis, defense_analysis)
        category_chart_images = generate_mirror_chart_images(detailed_analysis)
        touch_summary = _build_touch_summary(match_data)

        result = {
            'success': True,
            'radar_data': radar_data,
            'bout_type_stats': performance_data['bout_type_statistics'],
            'total_touches': performance_data['total_touches'],
            'detailed_analysis': detailed_analysis,
            'category_chart_images': category_chart_images,
            'touch_summary': touch_summary
        }

        # Sanitize all data to ensure JSON serializability  
        return sanitize_data_structure(result)
        
    except Exception as e:
        logging.error(f"Error generating basic video data: {e}")
        return {'success': False, 'error': str(e)}


def format_radar_data(metrics: Dict) -> Dict:
    """Format metrics for radar chart visualization"""
    
    # Order matches the axis labels in the specification
    radar_values = [
        sanitize_value(metrics['first_intention_effectiveness']),
        sanitize_value(metrics['second_intention_effectiveness']), 
        sanitize_value(metrics['attack_promptness']),
        sanitize_value(metrics['attack_aggressiveness']),
        sanitize_value(metrics['attack_distance_quality']),
        sanitize_value(metrics['attack_effectiveness']),
        sanitize_value(metrics['defense_distance_management']),
        sanitize_value(metrics['counter_execution_rate']),
        sanitize_value(metrics['defense_effectiveness'])
    ]
    
    return {
        'values': radar_values,
        'labels': [
            'First Intention Effectiveness',
            'Second Intention Effectiveness',
            'Attack Decisiveness',
            'Attack Intensity',
            'Attack Distance Quality',
            'Attack Success Rate',
            'Defense Distance Control',
            'Counter Execution Rate',
            'Defense Success Rate'
        ],
        'overall_score': sanitize_value(metrics['overall_score']),
        'subscores': {
            'in_box': sanitize_value(metrics['in_box_subscore']),
            'attack': sanitize_value(metrics['attack_subscore']) if metrics['attack_subscore'] is not None else None,
            'defense': sanitize_value(metrics['defense_subscore']) if metrics['defense_subscore'] is not None else None
        }
    }
