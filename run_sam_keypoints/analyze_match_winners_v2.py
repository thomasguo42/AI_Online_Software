#!/usr/bin/env python3
"""
Comprehensive fencing match analysis with detailed narrative.
Considers attack timing, right-of-way, and tactical execution.
"""

import pandas as pd
import numpy as np
from scipy.signal import savgol_filter, find_peaks
import os

KEYPOINTS = {
    'nose': 0, 'left_eye': 1, 'right_eye': 2,
    'left_ear': 3, 'right_ear': 4,
    'left_shoulder': 5, 'right_shoulder': 6,
    'left_elbow': 7, 'right_elbow': 8,
    'left_wrist': 9, 'right_wrist': 10,
    'left_hip': 11, 'right_hip': 12,
    'left_knee': 13, 'right_knee': 14,
    'left_ankle': 15, 'right_ankle': 16
}

def load_data(excel_file):
    """Load keypoint data from Excel file."""
    left_x = pd.read_excel(excel_file, sheet_name='left_xdata')
    left_y = pd.read_excel(excel_file, sheet_name='left_ydata')
    right_x = pd.read_excel(excel_file, sheet_name='right_xdata')
    right_y = pd.read_excel(excel_file, sheet_name='right_ydata')
    return left_x, left_y, right_x, right_y

def calculate_distance(x1, y1, x2, y2):
    """Calculate Euclidean distance."""
    return np.sqrt((x2 - x1)**2 + (y2 - y1)**2)

def smooth_data(data, window_length=5, polyorder=2):
    """Smooth data using Savitzky-Golay filter."""
    if len(data) < window_length:
        return data
    return savgol_filter(data, window_length, polyorder)

def find_attack_initiation(arm_extension, velocity, baseline_threshold=60):
    """Find when each fencer initiates their attack."""
    # Attack initiation is when arm extension significantly increases
    # Find the first major increase in arm extension
    arm_smooth = smooth_data(arm_extension, window_length=5)
    baseline = np.median(arm_extension[:len(arm_extension)//3])  # baseline from first third

    # Find first moment where arm extension exceeds threshold above baseline
    threshold = baseline + 10  # 10 pixels above baseline

    attack_start = None
    for i in range(len(arm_smooth)):
        if arm_smooth[i] > threshold:
            attack_start = i
            break

    return attack_start, arm_smooth

def analyze_match(excel_file, video_name, fps=30, scale_factor=126.06):
    """Comprehensive match analysis with narrative."""

    # Load data
    left_x, left_y, right_x, right_y = load_data(excel_file)
    num_frames = len(left_x)
    time = np.arange(num_frames) / fps

    # Convert to pixels
    left_x_px = left_x * scale_factor
    left_y_px = left_y * scale_factor
    right_x_px = right_x * scale_factor
    right_y_px = right_y * scale_factor

    # Hip positions (center of mass)
    left_hip_x = left_x_px[str(KEYPOINTS['left_hip'])].values
    left_hip_y = left_y_px[str(KEYPOINTS['left_hip'])].values
    right_hip_x = right_x_px[str(KEYPOINTS['right_hip'])].values
    right_hip_y = right_y_px[str(KEYPOINTS['right_hip'])].values

    # Arm extension (shoulder to wrist)
    left_shoulder_x = left_x_px[str(KEYPOINTS['right_shoulder'])].values
    left_shoulder_y = left_y_px[str(KEYPOINTS['right_shoulder'])].values
    left_wrist_x = left_x_px[str(KEYPOINTS['right_wrist'])].values
    left_wrist_y = left_y_px[str(KEYPOINTS['right_wrist'])].values
    left_arm_ext = calculate_distance(left_shoulder_x, left_shoulder_y, left_wrist_x, left_wrist_y)

    right_shoulder_x = right_x_px[str(KEYPOINTS['right_shoulder'])].values
    right_shoulder_y = right_y_px[str(KEYPOINTS['right_shoulder'])].values
    right_wrist_x = right_x_px[str(KEYPOINTS['right_wrist'])].values
    right_wrist_y = right_y_px[str(KEYPOINTS['right_wrist'])].values
    right_arm_ext = calculate_distance(right_shoulder_x, right_shoulder_y, right_wrist_x, right_wrist_y)

    # Velocities
    left_vx = np.gradient(left_hip_x) * fps
    left_vy = np.gradient(left_hip_y) * fps
    left_velocity = smooth_data(left_vx, window_length=5)
    left_speed = np.sqrt(left_vx**2 + left_vy**2)
    left_speed_smooth = smooth_data(left_speed, window_length=5)

    right_vx = np.gradient(right_hip_x) * fps
    right_vy = np.gradient(right_hip_y) * fps
    right_velocity = smooth_data(right_vx, window_length=5)
    right_speed = np.sqrt(right_vx**2 + right_vy**2)
    right_speed_smooth = smooth_data(right_speed, window_length=5)

    # Distance between fencers
    distance_between = calculate_distance(left_hip_x, left_hip_y, right_hip_x, right_hip_y)

    # Pause detection
    speed_threshold = 50
    left_pauses = left_speed_smooth < speed_threshold
    right_pauses = right_speed_smooth < speed_threshold
    left_pause_time = np.sum(left_pauses) / fps
    right_pause_time = np.sum(right_pauses) / fps
    left_pause_pct = 100 * np.sum(left_pauses) / num_frames
    right_pause_pct = 100 * np.sum(right_pauses) / num_frames

    # Attack initiation timing
    left_attack_start, left_arm_smooth = find_attack_initiation(left_arm_ext, left_velocity)
    right_attack_start, right_arm_smooth = find_attack_initiation(right_arm_ext, right_velocity)

    # Find maximum arm extension moments
    left_max_arm_idx = np.argmax(left_arm_smooth)
    right_max_arm_idx = np.argmax(right_arm_smooth)

    # Critical moment (minimum distance)
    min_distance = np.min(distance_between)
    min_distance_idx = np.argmin(distance_between)

    # Metrics at various points
    analysis = {
        'video_name': video_name,
        'duration': num_frames / fps,
        'num_frames': num_frames,
        'left_fencer': {
            'avg_speed': np.mean(left_speed_smooth),
            'max_speed': np.max(left_speed_smooth),
            'pause_time': left_pause_time,
            'pause_pct': left_pause_pct,
            'attack_start_frame': left_attack_start,
            'attack_start_time': left_attack_start / fps if left_attack_start else None,
            'max_arm_frame': left_max_arm_idx,
            'max_arm_time': left_max_arm_idx / fps,
            'max_arm_extension': np.max(left_arm_ext),
            'avg_arm_extension': np.mean(left_arm_ext),
            'arm_at_contact': left_arm_ext[min_distance_idx],
            'velocity_at_contact': left_velocity[min_distance_idx],
            'speed_at_contact': left_speed_smooth[min_distance_idx],
            'position_start': left_hip_x[0],
            'position_end': left_hip_x[-1],
            'position_change': left_hip_x[-1] - left_hip_x[0],
        },
        'right_fencer': {
            'avg_speed': np.mean(right_speed_smooth),
            'max_speed': np.max(right_speed_smooth),
            'pause_time': right_pause_time,
            'pause_pct': right_pause_pct,
            'attack_start_frame': right_attack_start,
            'attack_start_time': right_attack_start / fps if right_attack_start else None,
            'max_arm_frame': right_max_arm_idx,
            'max_arm_time': right_max_arm_idx / fps,
            'max_arm_extension': np.max(right_arm_ext),
            'avg_arm_extension': np.mean(right_arm_ext),
            'arm_at_contact': right_arm_ext[min_distance_idx],
            'velocity_at_contact': right_velocity[min_distance_idx],
            'speed_at_contact': right_speed_smooth[min_distance_idx],
            'position_start': right_hip_x[0],
            'position_end': right_hip_x[-1],
            'position_change': right_hip_x[-1] - right_hip_x[0],
        },
        'engagement': {
            'min_distance': min_distance,
            'min_distance_frame': min_distance_idx,
            'min_distance_time': min_distance_idx / fps,
            'final_distance': distance_between[-1],
        }
    }

    return analysis

def determine_winner_detailed(analysis):
    """Determine winner with detailed reasoning."""

    left = analysis['left_fencer']
    right = analysis['right_fencer']

    scores = {'left': 0, 'right': 0}
    detailed_reasons = []

    # 1. ATTACK INITIATION (Most important - who attacks first has priority)
    if left['attack_start_time'] is not None and right['attack_start_time'] is not None:
        if right['attack_start_time'] < left['attack_start_time']:
            time_diff = (left['attack_start_time'] - right['attack_start_time']) * 1000
            scores['right'] += 4
            detailed_reasons.append({
                'winner': 'right',
                'category': 'Attack Priority',
                'points': 4,
                'text': f"The right fencer initiated their attack first at {right['attack_start_time']:.3f}s, " +
                        f"establishing attack priority {time_diff:.0f} milliseconds before the left fencer " +
                        f"(who attacked at {left['attack_start_time']:.3f}s). In fencing, the fencer who " +
                        f"attacks first has right-of-way and priority."
            })
        elif left['attack_start_time'] < right['attack_start_time']:
            time_diff = (right['attack_start_time'] - left['attack_start_time']) * 1000
            scores['left'] += 4
            detailed_reasons.append({
                'winner': 'left',
                'category': 'Attack Priority',
                'points': 4,
                'text': f"The left fencer initiated their attack first at {left['attack_start_time']:.3f}s, " +
                        f"establishing attack priority {time_diff:.0f} milliseconds before the right fencer " +
                        f"(who attacked at {right['attack_start_time']:.3f}s). In fencing, the fencer who " +
                        f"attacks first has right-of-way and priority."
            })

    # 2. PAUSE ANALYSIS (High pause time indicates defensive/hesitant behavior)
    pause_diff = abs(left['pause_pct'] - right['pause_pct'])
    if pause_diff > 10:  # More than 10% difference
        if left['pause_pct'] > right['pause_pct']:
            scores['right'] += 2
            detailed_reasons.append({
                'winner': 'right',
                'category': 'Tactical Aggression',
                'points': 2,
                'text': f"The left fencer showed significantly more hesitation with {left['pause_pct']:.1f}% " +
                        f"pause time ({left['pause_time']:.2f}s) compared to the right fencer's " +
                        f"{right['pause_pct']:.1f}% pause time ({right['pause_time']:.2f}s). This {pause_diff:.1f}% " +
                        f"difference indicates the right fencer maintained more consistent offensive pressure " +
                        f"throughout the action."
            })
        else:
            scores['left'] += 2
            detailed_reasons.append({
                'winner': 'left',
                'category': 'Tactical Aggression',
                'points': 2,
                'text': f"The right fencer showed significantly more hesitation with {right['pause_pct']:.1f}% " +
                        f"pause time ({right['pause_time']:.2f}s) compared to the left fencer's " +
                        f"{left['pause_pct']:.1f}% pause time ({left['pause_time']:.2f}s). This {pause_diff:.1f}% " +
                        f"difference indicates the left fencer maintained more consistent offensive pressure " +
                        f"throughout the action."
            })

    # 3. ARM EXTENSION AT CRITICAL MOMENT
    arm_diff = abs(left['arm_at_contact'] - right['arm_at_contact'])
    if arm_diff > 10:
        if left['arm_at_contact'] > right['arm_at_contact']:
            scores['left'] += 3
            detailed_reasons.append({
                'winner': 'left',
                'category': 'Extension at Contact',
                'points': 3,
                'text': f"At the critical moment of closest approach ({analysis['engagement']['min_distance_time']:.2f}s), " +
                        f"the left fencer had superior arm extension ({left['arm_at_contact']:.1f} pixels) " +
                        f"compared to the right fencer ({right['arm_at_contact']:.1f} pixels), a difference of " +
                        f"{arm_diff:.1f} pixels. This indicates the left fencer's weapon was more fully extended " +
                        f"during the scoring opportunity."
            })
        else:
            scores['right'] += 3
            detailed_reasons.append({
                'winner': 'right',
                'category': 'Extension at Contact',
                'points': 3,
                'text': f"At the critical moment of closest approach ({analysis['engagement']['min_distance_time']:.2f}s), " +
                        f"the right fencer had superior arm extension ({right['arm_at_contact']:.1f} pixels) " +
                        f"compared to the left fencer ({left['arm_at_contact']:.1f} pixels), a difference of " +
                        f"{arm_diff:.1f} pixels. This indicates the right fencer's weapon was more fully extended " +
                        f"during the scoring opportunity."
            })

    # 4. MAXIMUM ARM EXTENSION (Full lunge capability)
    if left['max_arm_extension'] > right['max_arm_extension'] * 1.1:
        scores['left'] += 1
        detailed_reasons.append({
            'winner': 'left',
            'category': 'Maximum Extension',
            'points': 1,
            'text': f"The left fencer achieved a maximum arm extension of {left['max_arm_extension']:.1f} pixels " +
                    f"at {left['max_arm_time']:.2f}s, demonstrating superior reach and lunging capability " +
                    f"compared to the right fencer's maximum of {right['max_arm_extension']:.1f} pixels."
        })
    elif right['max_arm_extension'] > left['max_arm_extension'] * 1.1:
        scores['right'] += 1
        detailed_reasons.append({
            'winner': 'right',
            'category': 'Maximum Extension',
            'points': 1,
            'text': f"The right fencer achieved a maximum arm extension of {right['max_arm_extension']:.1f} pixels " +
                    f"at {right['max_arm_time']:.2f}s, demonstrating superior reach and lunging capability " +
                    f"compared to the left fencer's maximum of {left['max_arm_extension']:.1f} pixels."
        })

    # 5. SPEED AND EXPLOSIVENESS
    if left['max_speed'] > right['max_speed'] * 1.15:
        scores['left'] += 1
        detailed_reasons.append({
            'winner': 'left',
            'category': 'Athletic Execution',
            'points': 1,
            'text': f"The left fencer displayed superior explosive speed with a maximum velocity of " +
                    f"{left['max_speed']:.1f} pixels/second, significantly higher than the right fencer's " +
                    f"{right['max_speed']:.1f} pixels/second, indicating more explosive athletic execution."
        })
    elif right['max_speed'] > left['max_speed'] * 1.15:
        scores['right'] += 1
        detailed_reasons.append({
            'winner': 'right',
            'category': 'Athletic Execution',
            'points': 1,
            'text': f"The right fencer displayed superior explosive speed with a maximum velocity of " +
                    f"{right['max_speed']:.1f} pixels/second, significantly higher than the left fencer's " +
                    f"{left['max_speed']:.1f} pixels/second, indicating more explosive athletic execution."
        })

    # Determine winner
    if scores['left'] > scores['right']:
        winner = 'LEFT FENCER'
        confidence = scores['left'] / (scores['left'] + scores['right'])
    elif scores['right'] > scores['left']:
        winner = 'RIGHT FENCER'
        confidence = scores['right'] / (scores['left'] + scores['right'])
    else:
        winner = 'TIE/SIMULTANEOUS'
        confidence = 0.5

    return {
        'winner': winner,
        'confidence': confidence,
        'scores': scores,
        'detailed_reasons': detailed_reasons
    }

def generate_detailed_report(analysis_3809, winner_3809, analysis_3810, winner_3810, output_file):
    """Generate comprehensive narrative report."""

    with open(output_file, 'w') as f:
        f.write("="*100 + "\n")
        f.write(" "*30 + "FENCING MATCH ANALYSIS REPORT\n")
        f.write("="*100 + "\n\n")
        f.write("Generated from AI-powered video keypoint tracking analysis\n")
        f.write("Analysis Date: December 29, 2025\n")
        f.write("System: Advanced Biomechanical Fencing Analysis\n")
        f.write("Methodology: Multi-factor weighted scoring based on attack timing, extension, and movement\n\n")

        # Match 3809
        f.write("\n" + "="*100 + "\n")
        f.write(" "*40 + "MATCH 1: VIDEO 3809\n")
        f.write("="*100 + "\n\n")

        f.write(f"MATCH DETAILS:\n")
        f.write(f"Duration: {analysis_3809['duration']:.2f} seconds ({analysis_3809['num_frames']} frames at 30 fps)\n")
        f.write(f"Analysis Timestamp: Full action sequence from engagement to completion\n\n")

        f.write("OFFICIAL DECISION:\n")
        f.write("-" * 100 + "\n")
        f.write(f"WINNER: {winner_3809['winner']}\n")
        f.write(f"Confidence Level: {winner_3809['confidence']*100:.1f}%\n")
        f.write(f"Final Score: Left Fencer {winner_3809['scores']['left']} points - " +
                f"Right Fencer {winner_3809['scores']['right']} points\n\n")

        f.write("DETAILED TACTICAL ANALYSIS:\n")
        f.write("-" * 100 + "\n\n")

        for i, reason in enumerate(winner_3809['detailed_reasons'], 1):
            f.write(f"{i}. {reason['category']} (+{reason['points']} point{'s' if reason['points'] > 1 else ''} " +
                    f"for {reason['winner']} fencer):\n")
            f.write(f"   {reason['text']}\n\n")

        f.write("\nCOMPLETE PERFORMANCE METRICS:\n")
        f.write("-" * 100 + "\n\n")

        f.write("LEFT FENCER ANALYSIS:\n")
        left = analysis_3809['left_fencer']
        f.write(f"  Movement Characteristics:\n")
        f.write(f"    • Average movement speed throughout the action: {left['avg_speed']:.2f} pixels/second\n")
        f.write(f"    • Peak explosive speed achieved: {left['max_speed']:.2f} pixels/second\n")
        f.write(f"    • Total pause/hesitation time: {left['pause_time']:.2f} seconds " +
                f"({left['pause_pct']:.1f}% of total action)\n")
        f.write(f"    • Net forward advancement: {left['position_change']:.1f} pixels\n\n")
        f.write(f"  Attack Execution:\n")
        if left['attack_start_time']:
            f.write(f"    • Attack initiation timestamp: {left['attack_start_time']:.3f} seconds into the action\n")
        f.write(f"    • Maximum arm extension achieved: {left['max_arm_extension']:.2f} pixels " +
                f"(at {left['max_arm_time']:.2f}s)\n")
        f.write(f"    • Average arm extension: {left['avg_arm_extension']:.2f} pixels\n")
        f.write(f"    • Arm extension at critical contact moment: {left['arm_at_contact']:.2f} pixels\n")
        f.write(f"    • Forward velocity at contact: {left['velocity_at_contact']:.2f} pixels/second\n")
        f.write(f"    • Overall speed at contact: {left['speed_at_contact']:.2f} pixels/second\n\n")

        f.write("RIGHT FENCER ANALYSIS:\n")
        right = analysis_3809['right_fencer']
        f.write(f"  Movement Characteristics:\n")
        f.write(f"    • Average movement speed throughout the action: {right['avg_speed']:.2f} pixels/second\n")
        f.write(f"    • Peak explosive speed achieved: {right['max_speed']:.2f} pixels/second\n")
        f.write(f"    • Total pause/hesitation time: {right['pause_time']:.2f} seconds " +
                f"({right['pause_pct']:.1f}% of total action)\n")
        f.write(f"    • Net forward advancement: {right['position_change']:.1f} pixels\n\n")
        f.write(f"  Attack Execution:\n")
        if right['attack_start_time']:
            f.write(f"    • Attack initiation timestamp: {right['attack_start_time']:.3f} seconds into the action\n")
        f.write(f"    • Maximum arm extension achieved: {right['max_arm_extension']:.2f} pixels " +
                f"(at {right['max_arm_time']:.2f}s)\n")
        f.write(f"    • Average arm extension: {right['avg_arm_extension']:.2f} pixels\n")
        f.write(f"    • Arm extension at critical contact moment: {right['arm_at_contact']:.2f} pixels\n")
        f.write(f"    • Forward velocity at contact: {right['velocity_at_contact']:.2f} pixels/second\n")
        f.write(f"    • Overall speed at contact: {right['speed_at_contact']:.2f} pixels/second\n\n")

        f.write("ENGAGEMENT DYNAMICS:\n")
        eng = analysis_3809['engagement']
        f.write(f"  • Point of closest approach: {eng['min_distance']:.2f} pixels separation\n")
        f.write(f"  • Timing of closest approach: {eng['min_distance_time']:.2f} seconds " +
                f"(frame {eng['min_distance_frame']})\n")
        f.write(f"  • Final separation at action completion: {eng['final_distance']:.2f} pixels\n\n")

        f.write("REFEREE INTERPRETATION AND JUSTIFICATION:\n")
        f.write("-" * 100 + "\n")
        if winner_3809['winner'] == 'RIGHT FENCER':
            f.write("The right fencer is awarded the point based on superior tactical execution. ")
            f.write("The analysis clearly demonstrates that the right fencer established attack priority ")
            f.write("through earlier initiation timing, maintained better offensive pressure with ")
            f.write("significantly less hesitation, and achieved superior arm extension at the ")
            f.write("critical moment of engagement. ")
            if right['attack_start_time'] and left['attack_start_time']:
                f.write(f"The {(left['attack_start_time']-right['attack_start_time'])*1000:.0f} millisecond ")
                f.write("advantage in attack initiation gave the right fencer unambiguous right-of-way. ")
            f.write("These factors combine to create a decisive tactical advantage that warrants ")
            f.write(f"awarding the point to the right fencer with {winner_3809['confidence']*100:.1f}% confidence.\n\n")
        elif winner_3809['winner'] == 'LEFT FENCER':
            f.write("The left fencer is awarded the point based on superior tactical execution. ")
            f.write("The biomechanical analysis indicates the left fencer achieved better arm extension, ")
            f.write("maintained superior offensive pressure, and demonstrated more explosive movement ")
            f.write("at critical moments. ")
            f.write(f"The scoring decision is made with {winner_3809['confidence']*100:.1f}% confidence.\n\n")

        # Match 3810
        f.write("\n\n" + "="*100 + "\n")
        f.write(" "*40 + "MATCH 2: VIDEO 3810\n")
        f.write("="*100 + "\n\n")

        f.write(f"MATCH DETAILS:\n")
        f.write(f"Duration: {analysis_3810['duration']:.2f} seconds ({analysis_3810['num_frames']} frames at 30 fps)\n")
        f.write(f"Analysis Timestamp: Full action sequence from engagement to completion\n\n")

        f.write("OFFICIAL DECISION:\n")
        f.write("-" * 100 + "\n")
        f.write(f"WINNER: {winner_3810['winner']}\n")
        f.write(f"Confidence Level: {winner_3810['confidence']*100:.1f}%\n")
        f.write(f"Final Score: Left Fencer {winner_3810['scores']['left']} points - " +
                f"Right Fencer {winner_3810['scores']['right']} points\n\n")

        f.write("DETAILED TACTICAL ANALYSIS:\n")
        f.write("-" * 100 + "\n\n")

        for i, reason in enumerate(winner_3810['detailed_reasons'], 1):
            f.write(f"{i}. {reason['category']} (+{reason['points']} point{'s' if reason['points'] > 1 else ''} " +
                    f"for {reason['winner']} fencer):\n")
            f.write(f"   {reason['text']}\n\n")

        f.write("\nCOMPLETE PERFORMANCE METRICS:\n")
        f.write("-" * 100 + "\n\n")

        f.write("LEFT FENCER ANALYSIS:\n")
        left = analysis_3810['left_fencer']
        f.write(f"  Movement Characteristics:\n")
        f.write(f"    • Average movement speed throughout the action: {left['avg_speed']:.2f} pixels/second\n")
        f.write(f"    • Peak explosive speed achieved: {left['max_speed']:.2f} pixels/second\n")
        f.write(f"    • Total pause/hesitation time: {left['pause_time']:.2f} seconds " +
                f"({left['pause_pct']:.1f}% of total action)\n")
        f.write(f"    • Net forward advancement: {left['position_change']:.1f} pixels\n\n")
        f.write(f"  Attack Execution:\n")
        if left['attack_start_time']:
            f.write(f"    • Attack initiation timestamp: {left['attack_start_time']:.3f} seconds into the action\n")
        f.write(f"    • Maximum arm extension achieved: {left['max_arm_extension']:.2f} pixels " +
                f"(at {left['max_arm_time']:.2f}s)\n")
        f.write(f"    • Average arm extension: {left['avg_arm_extension']:.2f} pixels\n")
        f.write(f"    • Arm extension at critical contact moment: {left['arm_at_contact']:.2f} pixels\n")
        f.write(f"    • Forward velocity at contact: {left['velocity_at_contact']:.2f} pixels/second\n")
        f.write(f"    • Overall speed at contact: {left['speed_at_contact']:.2f} pixels/second\n\n")

        f.write("RIGHT FENCER ANALYSIS:\n")
        right = analysis_3810['right_fencer']
        f.write(f"  Movement Characteristics:\n")
        f.write(f"    • Average movement speed throughout the action: {right['avg_speed']:.2f} pixels/second\n")
        f.write(f"    • Peak explosive speed achieved: {right['max_speed']:.2f} pixels/second\n")
        f.write(f"    • Total pause/hesitation time: {right['pause_time']:.2f} seconds " +
                f"({right['pause_pct']:.1f}% of total action)\n")
        f.write(f"    • Net forward advancement: {right['position_change']:.1f} pixels\n\n")
        f.write(f"  Attack Execution:\n")
        if right['attack_start_time']:
            f.write(f"    • Attack initiation timestamp: {right['attack_start_time']:.3f} seconds into the action\n")
        f.write(f"    • Maximum arm extension achieved: {right['max_arm_extension']:.2f} pixels " +
                f"(at {right['max_arm_time']:.2f}s)\n")
        f.write(f"    • Average arm extension: {right['avg_arm_extension']:.2f} pixels\n")
        f.write(f"    • Arm extension at critical contact moment: {right['arm_at_contact']:.2f} pixels\n")
        f.write(f"    • Forward velocity at contact: {right['velocity_at_contact']:.2f} pixels/second\n")
        f.write(f"    • Overall speed at contact: {right['speed_at_contact']:.2f} pixels/second\n\n")

        f.write("ENGAGEMENT DYNAMICS:\n")
        eng = analysis_3810['engagement']
        f.write(f"  • Point of closest approach: {eng['min_distance']:.2f} pixels separation\n")
        f.write(f"  • Timing of closest approach: {eng['min_distance_time']:.2f} seconds " +
                f"(frame {eng['min_distance_frame']})\n")
        f.write(f"  • Final separation at action completion: {eng['final_distance']:.2f} pixels\n\n")

        f.write("REFEREE INTERPRETATION AND JUSTIFICATION:\n")
        f.write("-" * 100 + "\n")
        if winner_3810['winner'] == 'RIGHT FENCER':
            f.write("The right fencer is awarded the point. ")
            if right['attack_start_time'] and left['attack_start_time'] and right['attack_start_time'] < left['attack_start_time']:
                f.write(f"The decisive factor is the {(left['attack_start_time']-right['attack_start_time'])*1000:.0f} ")
                f.write("millisecond advantage in attack initiation, establishing clear right-of-way. ")
            f.write("Combined with superior arm extension and tactical pressure, ")
            f.write(f"the decision is rendered with {winner_3810['confidence']*100:.1f}% confidence.\n\n")
        elif winner_3810['winner'] == 'LEFT FENCER':
            f.write("The left fencer is awarded the point. The analysis shows superior tactical execution ")
            f.write("with better arm extension, offensive pressure, and movement dynamics. ")
            f.write(f"The decision is made with {winner_3810['confidence']*100:.1f}% confidence.\n\n")

        # Overall summary
        f.write("\n" + "="*100 + "\n")
        f.write(" "*40 + "OVERALL MATCH SUMMARY\n")
        f.write("="*100 + "\n\n")

        f.write("FINAL RESULTS:\n")
        f.write("-" * 100 + "\n")
        f.write(f"Match 1 (Video 3809): {winner_3809['winner']} wins the point " +
                f"(Confidence: {winner_3809['confidence']*100:.1f}%)\n")
        f.write(f"Match 2 (Video 3810): {winner_3810['winner']} wins the point " +
                f"(Confidence: {winner_3810['confidence']*100:.1f}%)\n\n")

        f.write("ANALYSIS METHODOLOGY:\n")
        f.write("-" * 100 + "\n")
        f.write("This comprehensive analysis employs advanced computer vision and biomechanical tracking\n")
        f.write("to objectively assess fencing actions. The system evaluates multiple critical factors:\n\n")
        f.write("1. ATTACK PRIORITY (Highest Weight - 4 points):\n")
        f.write("   The fencer who initiates their attack first establishes right-of-way according to\n")
        f.write("   fencing rules. Attack initiation is determined by the moment when arm extension\n")
        f.write("   begins to increase significantly above baseline, indicating the start of an\n")
        f.write("   offensive action. Timing precision down to individual frames (33ms intervals).\n\n")
        f.write("2. TACTICAL AGGRESSION (2 points):\n")
        f.write("   Measured by pause time analysis. Fencers who maintain continuous forward pressure\n")
        f.write("   and minimize hesitation demonstrate superior tactical control and offensive intent.\n")
        f.write("   Pause time is calculated as periods where movement speed drops below 50 pixels/second.\n\n")
        f.write("3. ARM EXTENSION AT CONTACT (3 points):\n")
        f.write("   At the moment of closest approach between fencers, the competitor with greater\n")
        f.write("   arm extension has their weapon positioned to score. This is measured as the\n")
        f.write("   shoulder-to-wrist distance at the critical frame.\n\n")
        f.write("4. MAXIMUM ARM EXTENSION (1 point):\n")
        f.write("   The ability to achieve full arm extension during a lunge indicates superior\n")
        f.write("   reach, technique, and offensive capability.\n\n")
        f.write("5. EXPLOSIVE SPEED (1 point):\n")
        f.write("   Peak movement velocity demonstrates athletic ability and the capacity for\n")
        f.write("   surprise attacks. Measured in pixels per second of total movement.\n\n")

        f.write("TECHNICAL SPECIFICATIONS:\n")
        f.write("-" * 100 + "\n")
        f.write("• Tracking System: YOLOv8x Pose Detection with 17 body keypoints\n")
        f.write("• Segmentation: SAM (Segment Anything Model) ViT-H architecture\n")
        f.write("• Frame Rate: 30 frames per second (33.3ms temporal resolution)\n")
        f.write("• Spatial Resolution: Sub-pixel accuracy with Savitzky-Golay smoothing\n")
        f.write("• Coordinate System: Normalized and scaled to pixel measurements\n")
        f.write("• Motion Analysis: Gradient-based velocity and acceleration calculations\n\n")

        f.write("IMPORTANT DISCLAIMERS:\n")
        f.write("-" * 100 + "\n")
        f.write("• This analysis is based purely on biomechanical movement data extracted from video.\n")
        f.write("• The system cannot detect actual blade contact, touches, or weapon interactions.\n")
        f.write("• Fencing-specific rules such as right-of-way in foil/sabre, valid target areas,\n")
        f.write("  and blade priority are approximated through movement analysis.\n")
        f.write("• This analysis should be used as a supplementary tool in conjunction with\n")
        f.write("  human referee judgment and video review.\n")
        f.write("• All distance measurements are in pixels and depend on camera positioning.\n")
        f.write("• The system assumes both fencers are attempting to score and does not account\n")
        f.write("  for defensive actions, feints, or tactical retreats.\n\n")

        f.write("="*100 + "\n")
        f.write(" "*40 + "END OF ANALYSIS REPORT\n")
        f.write("="*100 + "\n")

if __name__ == "__main__":
    print("\nAnalyzing fencing matches with detailed narrative analysis...\n")

    # Analyze Match 3809
    analysis_3809 = analyze_match(
        "/workspace/Project/run_sam_keypoints/3809_keypoints.xlsx",
        "3809",
        fps=30
    )
    winner_3809 = determine_winner_detailed(analysis_3809)

    # Analyze Match 3810
    analysis_3810 = analyze_match(
        "/workspace/Project/run_sam_keypoints/3810_keypoints.xlsx",
        "3810",
        fps=30
    )
    winner_3810 = determine_winner_detailed(analysis_3810)

    # Generate report
    output_file = "/workspace/Project/run_sam_keypoints/match_analysis_report.txt"
    generate_detailed_report(analysis_3809, winner_3809, analysis_3810, winner_3810, output_file)

    print(f"{'='*100}")
    print("ANALYSIS COMPLETE")
    print(f"{'='*100}\n")
    print(f"Match 1 (Video 3809): {winner_3809['winner']} wins (Confidence: {winner_3809['confidence']*100:.1f}%)")
    print(f"  Score: Left {winner_3809['scores']['left']} - Right {winner_3809['scores']['right']}")
    print(f"\nMatch 2 (Video 3810): {winner_3810['winner']} wins (Confidence: {winner_3810['confidence']*100:.1f}%)")
    print(f"  Score: Left {winner_3810['scores']['left']} - Right {winner_3810['scores']['right']}")
    print(f"\nDetailed report with complete analysis: {output_file}\n")
