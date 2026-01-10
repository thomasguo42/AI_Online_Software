#!/usr/bin/env python3
"""
Analyze fencing matches to determine winners based on movement data.
"""

import pandas as pd
import numpy as np
from scipy.signal import savgol_filter
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

def detect_attacks(arm_extension, velocity, threshold_percentile=75):
    """Detect attack moments based on arm extension and velocity."""
    # Attacks are characterized by:
    # 1. High arm extension (lunging)
    # 2. Forward velocity
    arm_threshold = np.percentile(arm_extension, threshold_percentile)
    attacks = arm_extension > arm_threshold
    return attacks

def analyze_match(excel_file, video_name, fps=30, scale_factor=126.06):
    """Comprehensive match analysis."""

    print(f"\n{'='*80}")
    print(f"ANALYZING: {video_name}")
    print(f"{'='*80}\n")

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

    # Detect attacks
    left_attacks = detect_attacks(left_arm_ext, left_velocity)
    right_attacks = detect_attacks(right_arm_ext, right_velocity)

    # Movement direction analysis (positive = moving right, negative = moving left)
    left_forward_movement = np.sum(left_velocity > 50)  # Moving right significantly
    left_backward_movement = np.sum(left_velocity < -50)  # Moving left significantly
    right_forward_movement = np.sum(right_velocity < -50)  # Moving left significantly (towards left fencer)
    right_backward_movement = np.sum(right_velocity > 50)  # Moving right significantly

    # Distance closing analysis
    distance_change = distance_between[-1] - distance_between[0]
    distance_closing_rate = -np.gradient(distance_between).mean()

    # Final positions
    left_position_change = left_hip_x[-1] - left_hip_x[0]
    right_position_change = right_hip_x[-1] - right_hip_x[0]

    # Attack analysis
    left_attack_frames = np.sum(left_attacks)
    right_attack_frames = np.sum(right_attacks)

    # Speed and aggression metrics
    left_avg_speed = np.mean(left_speed_smooth)
    right_avg_speed = np.mean(right_speed_smooth)
    left_max_speed = np.max(left_speed_smooth)
    right_max_speed = np.max(right_speed_smooth)

    # Arm extension analysis
    left_max_arm_ext = np.max(left_arm_ext)
    right_max_arm_ext = np.max(right_arm_ext)
    left_avg_arm_ext = np.mean(left_arm_ext)
    right_avg_arm_ext = np.mean(right_arm_ext)

    # Final distance analysis (closer = likely scored)
    final_distance = distance_between[-1]
    min_distance = np.min(distance_between)
    min_distance_idx = np.argmin(distance_between)

    # Determine who closed distance at critical moment
    if min_distance_idx < len(distance_between) / 2:
        critical_phase = "early"
    else:
        critical_phase = "late"

    # Velocity at minimum distance (who was advancing)
    left_vel_at_min = left_velocity[min_distance_idx]
    right_vel_at_min = right_velocity[min_distance_idx]

    # Arm extension at minimum distance (who was attacking)
    left_arm_at_min = left_arm_ext[min_distance_idx]
    right_arm_at_min = right_arm_ext[min_distance_idx]

    # Analysis dictionary
    analysis = {
        'video_name': video_name,
        'duration': num_frames / fps,
        'left_fencer': {
            'avg_speed': left_avg_speed,
            'max_speed': left_max_speed,
            'forward_frames': left_forward_movement,
            'backward_frames': left_backward_movement,
            'attack_frames': left_attack_frames,
            'position_change': left_position_change,
            'max_arm_extension': left_max_arm_ext,
            'avg_arm_extension': left_avg_arm_ext,
            'velocity_at_contact': left_vel_at_min,
            'arm_ext_at_contact': left_arm_at_min
        },
        'right_fencer': {
            'avg_speed': right_avg_speed,
            'max_speed': right_max_speed,
            'forward_frames': right_forward_movement,
            'backward_frames': right_backward_movement,
            'attack_frames': right_attack_frames,
            'position_change': right_position_change,
            'max_arm_extension': right_max_arm_ext,
            'avg_arm_extension': right_avg_arm_ext,
            'velocity_at_contact': right_vel_at_min,
            'arm_ext_at_contact': right_arm_at_min
        },
        'engagement': {
            'min_distance': min_distance,
            'final_distance': final_distance,
            'distance_change': distance_change,
            'critical_phase': critical_phase,
            'min_distance_time': min_distance_idx / fps
        }
    }

    return analysis

def determine_winner(analysis):
    """Determine match winner based on analysis."""

    left = analysis['left_fencer']
    right = analysis['right_fencer']
    engagement = analysis['engagement']

    # Scoring criteria (weighted)
    scores = {'left': 0, 'right': 0}
    reasons = {'left': [], 'right': []}

    # 1. Attack initiation (arm extension at critical moment)
    if left['arm_ext_at_contact'] > right['arm_ext_at_contact'] * 1.1:
        scores['left'] += 3
        reasons['left'].append(f"Extended arm at critical moment ({left['arm_ext_at_contact']:.1f}px vs {right['arm_ext_at_contact']:.1f}px)")
    elif right['arm_ext_at_contact'] > left['arm_ext_at_contact'] * 1.1:
        scores['right'] += 3
        reasons['right'].append(f"Extended arm at critical moment ({right['arm_ext_at_contact']:.1f}px vs {left['arm_ext_at_contact']:.1f}px)")

    # 2. Forward velocity at contact (advancing/attacking)
    # Left fencer moving right is positive, right fencer moving left is negative
    left_advancing = left['velocity_at_contact'] > 0
    right_advancing = right['velocity_at_contact'] < 0

    if left_advancing and abs(left['velocity_at_contact']) > abs(right['velocity_at_contact']) * 1.2:
        scores['left'] += 2
        reasons['left'].append(f"Advancing at contact ({left['velocity_at_contact']:.1f} px/s)")
    elif right_advancing and abs(right['velocity_at_contact']) > abs(left['velocity_at_contact']) * 1.2:
        scores['right'] += 2
        reasons['right'].append(f"Advancing at contact ({abs(right['velocity_at_contact']):.1f} px/s)")

    # 3. Overall attack frequency
    if left['attack_frames'] > right['attack_frames'] * 1.2:
        scores['left'] += 2
        reasons['left'].append(f"More offensive action ({left['attack_frames']} frames vs {right['attack_frames']} frames)")
    elif right['attack_frames'] > left['attack_frames'] * 1.2:
        scores['right'] += 2
        reasons['right'].append(f"More offensive action ({right['attack_frames']} frames vs {left['attack_frames']} frames)")

    # 4. Forward movement dominance
    left_net_forward = left['forward_frames'] - left['backward_frames']
    right_net_forward = right['forward_frames'] - right['backward_frames']

    if left_net_forward > right_net_forward + 5:
        scores['left'] += 1
        reasons['left'].append(f"Greater forward pressure (net {left_net_forward} frames)")
    elif right_net_forward > left_net_forward + 5:
        scores['right'] += 1
        reasons['right'].append(f"Greater forward pressure (net {right_net_forward} frames)")

    # 5. Maximum arm extension (full lunge)
    if left['max_arm_extension'] > right['max_arm_extension'] * 1.15:
        scores['left'] += 1
        reasons['left'].append(f"Superior lunge extension ({left['max_arm_extension']:.1f}px)")
    elif right['max_arm_extension'] > left['max_arm_extension'] * 1.15:
        scores['right'] += 1
        reasons['right'].append(f"Superior lunge extension ({right['max_arm_extension']:.1f}px)")

    # 6. Speed and explosiveness
    if left['max_speed'] > right['max_speed'] * 1.2:
        scores['left'] += 1
        reasons['left'].append(f"Higher explosive speed ({left['max_speed']:.1f} px/s)")
    elif right['max_speed'] > left['max_speed'] * 1.2:
        scores['right'] += 1
        reasons['right'].append(f"Higher explosive speed ({right['max_speed']:.1f} px/s)")

    # Determine winner
    if scores['left'] > scores['right']:
        winner = 'LEFT FENCER'
        winner_reasons = reasons['left']
        confidence = scores['left'] / (scores['left'] + scores['right'])
    elif scores['right'] > scores['left']:
        winner = 'RIGHT FENCER'
        winner_reasons = reasons['right']
        confidence = scores['right'] / (scores['left'] + scores['right'])
    else:
        winner = 'TIE/UNCLEAR'
        winner_reasons = ['Equal tactical execution']
        confidence = 0.5

    return {
        'winner': winner,
        'confidence': confidence,
        'scores': scores,
        'reasons': winner_reasons,
        'all_reasons': reasons
    }

def generate_report(analysis_3809, winner_3809, analysis_3810, winner_3810, output_file):
    """Generate comprehensive text report."""

    with open(output_file, 'w') as f:
        f.write("="*80 + "\n")
        f.write("FENCING MATCH ANALYSIS REPORT\n")
        f.write("="*80 + "\n\n")
        f.write("Generated from video keypoint tracking data\n")
        f.write("Analysis Date: 2025-12-29\n")
        f.write("Analyst: AI-Powered Fencing Analysis System\n\n")

        # Match 3809
        f.write("\n" + "="*80 + "\n")
        f.write("MATCH 1: Video 3809\n")
        f.write("="*80 + "\n\n")

        f.write(f"Duration: {analysis_3809['duration']:.2f} seconds ({len(analysis_3809['left_fencer'])} frames)\n\n")

        f.write("MATCH SUMMARY:\n")
        f.write("-" * 80 + "\n")
        f.write(f"Winner: {winner_3809['winner']}\n")
        f.write(f"Confidence: {winner_3809['confidence']*100:.1f}%\n")
        f.write(f"Score: Left Fencer {winner_3809['scores']['left']} - Right Fencer {winner_3809['scores']['right']}\n\n")

        f.write("WINNING REASONS:\n")
        for i, reason in enumerate(winner_3809['reasons'], 1):
            f.write(f"  {i}. {reason}\n")
        f.write("\n")

        f.write("DETAILED ANALYSIS:\n")
        f.write("-" * 80 + "\n\n")

        f.write("Left Fencer:\n")
        left = analysis_3809['left_fencer']
        f.write(f"  • Average Speed: {left['avg_speed']:.2f} px/s\n")
        f.write(f"  • Maximum Speed: {left['max_speed']:.2f} px/s\n")
        f.write(f"  • Forward Movement: {left['forward_frames']} frames\n")
        f.write(f"  • Backward Movement: {left['backward_frames']} frames\n")
        f.write(f"  • Attack Frames: {left['attack_frames']} frames\n")
        f.write(f"  • Maximum Arm Extension: {left['max_arm_extension']:.2f} px\n")
        f.write(f"  • Average Arm Extension: {left['avg_arm_extension']:.2f} px\n")
        f.write(f"  • Velocity at Critical Moment: {left['velocity_at_contact']:.2f} px/s\n")
        f.write(f"  • Arm Extension at Critical Moment: {left['arm_ext_at_contact']:.2f} px\n\n")

        f.write("Right Fencer:\n")
        right = analysis_3809['right_fencer']
        f.write(f"  • Average Speed: {right['avg_speed']:.2f} px/s\n")
        f.write(f"  • Maximum Speed: {right['max_speed']:.2f} px/s\n")
        f.write(f"  • Forward Movement: {right['forward_frames']} frames\n")
        f.write(f"  • Backward Movement: {right['backward_frames']} frames\n")
        f.write(f"  • Attack Frames: {right['attack_frames']} frames\n")
        f.write(f"  • Maximum Arm Extension: {right['max_arm_extension']:.2f} px\n")
        f.write(f"  • Average Arm Extension: {right['avg_arm_extension']:.2f} px\n")
        f.write(f"  • Velocity at Critical Moment: {right['velocity_at_contact']:.2f} px/s\n")
        f.write(f"  • Arm Extension at Critical Moment: {right['arm_ext_at_contact']:.2f} px\n\n")

        f.write("Engagement Dynamics:\n")
        eng = analysis_3809['engagement']
        f.write(f"  • Minimum Distance: {eng['min_distance']:.2f} px\n")
        f.write(f"  • Critical Phase: {eng['critical_phase']} in the action\n")
        f.write(f"  • Time of Closest Approach: {eng['min_distance_time']:.2f} seconds\n")
        f.write(f"  • Final Distance: {eng['final_distance']:.2f} px\n\n")

        f.write("TACTICAL INTERPRETATION:\n")
        f.write("-" * 80 + "\n")
        if winner_3809['winner'] == 'LEFT FENCER':
            f.write("The left fencer demonstrated superior offensive tactics with better\n")
            f.write("timing, extension, and forward pressure at the critical moment.\n")
        elif winner_3809['winner'] == 'RIGHT FENCER':
            f.write("The right fencer showed dominant attacking action with superior\n")
            f.write("arm extension and forward movement at the decisive moment.\n")
        else:
            f.write("Both fencers demonstrated equal tactical skill. A more detailed\n")
            f.write("review or additional footage would be needed for a definitive call.\n")

        # Match 3810
        f.write("\n\n" + "="*80 + "\n")
        f.write("MATCH 2: Video 3810\n")
        f.write("="*80 + "\n\n")

        f.write(f"Duration: {analysis_3810['duration']:.2f} seconds\n\n")

        f.write("MATCH SUMMARY:\n")
        f.write("-" * 80 + "\n")
        f.write(f"Winner: {winner_3810['winner']}\n")
        f.write(f"Confidence: {winner_3810['confidence']*100:.1f}%\n")
        f.write(f"Score: Left Fencer {winner_3810['scores']['left']} - Right Fencer {winner_3810['scores']['right']}\n\n")

        f.write("WINNING REASONS:\n")
        for i, reason in enumerate(winner_3810['reasons'], 1):
            f.write(f"  {i}. {reason}\n")
        f.write("\n")

        f.write("DETAILED ANALYSIS:\n")
        f.write("-" * 80 + "\n\n")

        f.write("Left Fencer:\n")
        left = analysis_3810['left_fencer']
        f.write(f"  • Average Speed: {left['avg_speed']:.2f} px/s\n")
        f.write(f"  • Maximum Speed: {left['max_speed']:.2f} px/s\n")
        f.write(f"  • Forward Movement: {left['forward_frames']} frames\n")
        f.write(f"  • Backward Movement: {left['backward_frames']} frames\n")
        f.write(f"  • Attack Frames: {left['attack_frames']} frames\n")
        f.write(f"  • Maximum Arm Extension: {left['max_arm_extension']:.2f} px\n")
        f.write(f"  • Average Arm Extension: {left['avg_arm_extension']:.2f} px\n")
        f.write(f"  • Velocity at Critical Moment: {left['velocity_at_contact']:.2f} px/s\n")
        f.write(f"  • Arm Extension at Critical Moment: {left['arm_ext_at_contact']:.2f} px\n\n")

        f.write("Right Fencer:\n")
        right = analysis_3810['right_fencer']
        f.write(f"  • Average Speed: {right['avg_speed']:.2f} px/s\n")
        f.write(f"  • Maximum Speed: {right['max_speed']:.2f} px/s\n")
        f.write(f"  • Forward Movement: {right['forward_frames']} frames\n")
        f.write(f"  • Backward Movement: {right['backward_frames']} frames\n")
        f.write(f"  • Attack Frames: {right['attack_frames']} frames\n")
        f.write(f"  • Maximum Arm Extension: {right['max_arm_extension']:.2f} px\n")
        f.write(f"  • Average Arm Extension: {right['avg_arm_extension']:.2f} px\n")
        f.write(f"  • Velocity at Critical Moment: {right['velocity_at_contact']:.2f} px/s\n")
        f.write(f"  • Arm Extension at Critical Moment: {right['arm_ext_at_contact']:.2f} px\n\n")

        f.write("Engagement Dynamics:\n")
        eng = analysis_3810['engagement']
        f.write(f"  • Minimum Distance: {eng['min_distance']:.2f} px\n")
        f.write(f"  • Critical Phase: {eng['critical_phase']} in the action\n")
        f.write(f"  • Time of Closest Approach: {eng['min_distance_time']:.2f} seconds\n")
        f.write(f"  • Final Distance: {eng['final_distance']:.2f} px\n\n")

        f.write("TACTICAL INTERPRETATION:\n")
        f.write("-" * 80 + "\n")
        if winner_3810['winner'] == 'LEFT FENCER':
            f.write("The left fencer executed a successful attack with better timing,\n")
            f.write("arm extension, and offensive pressure at the decisive moment.\n")
        elif winner_3810['winner'] == 'RIGHT FENCER':
            f.write("The right fencer controlled the action with superior attacking\n")
            f.write("movements and arm extension at the critical moment of engagement.\n")
        else:
            f.write("Both fencers showed comparable tactical execution. Additional\n")
            f.write("analysis or referee judgment would be required for a final call.\n")

        # Overall summary
        f.write("\n\n" + "="*80 + "\n")
        f.write("OVERALL ASSESSMENT\n")
        f.write("="*80 + "\n\n")

        f.write("Match 1 (3809): " + winner_3809['winner'] + f" wins with {winner_3809['confidence']*100:.1f}% confidence\n")
        f.write("Match 2 (3810): " + winner_3810['winner'] + f" wins with {winner_3810['confidence']*100:.1f}% confidence\n\n")

        f.write("METHODOLOGY:\n")
        f.write("-" * 80 + "\n")
        f.write("This analysis is based on biomechanical tracking of fencers' movements,\n")
        f.write("including:\n")
        f.write("  • Arm extension (indicates attacking action/lunges)\n")
        f.write("  • Forward/backward velocity (indicates offensive/defensive intent)\n")
        f.write("  • Distance management (control of engagement distance)\n")
        f.write("  • Attack frequency (offensive pressure)\n")
        f.write("  • Speed and explosiveness (athletic execution)\n\n")
        f.write("The winner is determined by weighted scoring of these factors,\n")
        f.write("with emphasis on actions at the critical moment of closest approach.\n\n")

        f.write("NOTES:\n")
        f.write("-" * 80 + "\n")
        f.write("• This analysis is based on movement data only and does not account\n")
        f.write("  for blade contact, right-of-way rules, or other fencing-specific\n")
        f.write("  factors that a human referee would consider.\n")
        f.write("• Results should be used in conjunction with video review and\n")
        f.write("  referee judgment for official scoring.\n")
        f.write("• All measurements are in pixels; actual distances depend on\n")
        f.write("  camera angle and distance from subjects.\n\n")

        f.write("="*80 + "\n")
        f.write("END OF REPORT\n")
        f.write("="*80 + "\n")

if __name__ == "__main__":
    # Analyze Match 3809
    analysis_3809 = analyze_match(
        "/workspace/Project/run_sam_keypoints/3809_keypoints.xlsx",
        "3809",
        fps=30
    )
    winner_3809 = determine_winner(analysis_3809)

    # Analyze Match 3810
    analysis_3810 = analyze_match(
        "/workspace/Project/run_sam_keypoints/3810_keypoints.xlsx",
        "3810",
        fps=30
    )
    winner_3810 = determine_winner(analysis_3810)

    # Generate report
    output_file = "/workspace/Project/run_sam_keypoints/match_analysis_report.txt"
    generate_report(analysis_3809, winner_3809, analysis_3810, winner_3810, output_file)

    print(f"\n{'='*80}")
    print("ANALYSIS COMPLETE")
    print(f"{'='*80}\n")
    print(f"Match 3809: {winner_3809['winner']} wins ({winner_3809['confidence']*100:.1f}% confidence)")
    print(f"Match 3810: {winner_3810['winner']} wins ({winner_3810['confidence']*100:.1f}% confidence)")
    print(f"\nDetailed report saved to: {output_file}\n")
