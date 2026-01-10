#!/usr/bin/env python3
"""
Generate comprehensive analysis graphs for fencing videos.
Analyzes movement patterns, velocity, acceleration, arm extension, and more.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy.signal import savgol_filter
import os

# YOLO Pose keypoint indices
KEYPOINTS = {
    'nose': 0,
    'left_eye': 1, 'right_eye': 2,
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

def calculate_velocity(positions, fps=30):
    """Calculate velocity from position data."""
    velocity = np.gradient(positions, axis=0) * fps
    return velocity

def calculate_acceleration(velocity, fps=30):
    """Calculate acceleration from velocity data."""
    acceleration = np.gradient(velocity, axis=0) * fps
    return acceleration

def calculate_distance(x1, y1, x2, y2):
    """Calculate Euclidean distance between two points."""
    return np.sqrt((x2 - x1)**2 + (y2 - y1)**2)

def smooth_data(data, window_length=5, polyorder=2):
    """Smooth data using Savitzky-Golay filter."""
    if len(data) < window_length:
        return data
    return savgol_filter(data, window_length, polyorder)

def analyze_video(excel_file, output_prefix, fps=30, scale_factor=126.06):
    """Generate comprehensive analysis graphs for a video."""

    print(f"\nAnalyzing {os.path.basename(excel_file)}...")

    # Load data
    left_x, left_y, right_x, right_y = load_data(excel_file)
    num_frames = len(left_x)
    time = np.arange(num_frames) / fps

    # Convert to pixel coordinates
    left_x_px = left_x * scale_factor
    left_y_px = left_y * scale_factor
    right_x_px = right_x * scale_factor
    right_y_px = right_y * scale_factor

    # Create figure with multiple subplots
    fig = plt.figure(figsize=(20, 24))
    gs = gridspec.GridSpec(6, 2, figure=fig, hspace=0.3, wspace=0.3)

    # === 1. X-Coordinate Movement Over Time ===
    ax1 = fig.add_subplot(gs[0, :])
    ax1.plot(time, left_x_px[str(KEYPOINTS['left_hip'])], 'b-', label='Left Fencer Hip', linewidth=2)
    ax1.plot(time, right_x_px[str(KEYPOINTS['right_hip'])], 'r-', label='Right Fencer Hip', linewidth=2)
    ax1.set_xlabel('Time (seconds)', fontsize=12)
    ax1.set_ylabel('X Position (pixels)', fontsize=12)
    ax1.set_title('Horizontal Movement Over Time (Hip Position)', fontsize=14, fontweight='bold')
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)

    # === 2. Velocity Analysis ===
    ax2 = fig.add_subplot(gs[1, 0])
    left_hip_x = left_x_px[str(KEYPOINTS['left_hip'])].values
    right_hip_x = right_x_px[str(KEYPOINTS['right_hip'])].values

    left_velocity = np.gradient(left_hip_x) * fps
    right_velocity = np.gradient(right_hip_x) * fps

    left_velocity_smooth = smooth_data(left_velocity, window_length=5)
    right_velocity_smooth = smooth_data(right_velocity, window_length=5)

    ax2.plot(time, left_velocity_smooth, 'b-', label='Left Fencer', linewidth=2)
    ax2.plot(time, right_velocity_smooth, 'r-', label='Right Fencer', linewidth=2)
    ax2.axhline(y=0, color='k', linestyle='--', alpha=0.3)
    ax2.set_xlabel('Time (seconds)', fontsize=12)
    ax2.set_ylabel('Velocity (pixels/sec)', fontsize=12)
    ax2.set_title('Horizontal Velocity', fontsize=14, fontweight='bold')
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)

    # === 3. Acceleration Analysis ===
    ax3 = fig.add_subplot(gs[1, 1])
    left_accel = np.gradient(left_velocity_smooth) * fps
    right_accel = np.gradient(right_velocity_smooth) * fps

    ax3.plot(time, left_accel, 'b-', label='Left Fencer', linewidth=2)
    ax3.plot(time, right_accel, 'r-', label='Right Fencer', linewidth=2)
    ax3.axhline(y=0, color='k', linestyle='--', alpha=0.3)
    ax3.set_xlabel('Time (seconds)', fontsize=12)
    ax3.set_ylabel('Acceleration (pixels/sec²)', fontsize=12)
    ax3.set_title('Horizontal Acceleration', fontsize=14, fontweight='bold')
    ax3.legend(fontsize=10)
    ax3.grid(True, alpha=0.3)

    # === 4. Arm Extension (Shoulder to Wrist Distance) ===
    ax4 = fig.add_subplot(gs[2, 0])

    # Left fencer - right arm (sword arm for right-handed)
    left_shoulder_x = left_x_px[str(KEYPOINTS['right_shoulder'])].values
    left_shoulder_y = left_y_px[str(KEYPOINTS['right_shoulder'])].values
    left_wrist_x = left_x_px[str(KEYPOINTS['right_wrist'])].values
    left_wrist_y = left_y_px[str(KEYPOINTS['right_wrist'])].values
    left_arm_ext = calculate_distance(left_shoulder_x, left_shoulder_y, left_wrist_x, left_wrist_y)

    # Right fencer - right arm
    right_shoulder_x = right_x_px[str(KEYPOINTS['right_shoulder'])].values
    right_shoulder_y = right_y_px[str(KEYPOINTS['right_shoulder'])].values
    right_wrist_x = right_x_px[str(KEYPOINTS['right_wrist'])].values
    right_wrist_y = right_y_px[str(KEYPOINTS['right_wrist'])].values
    right_arm_ext = calculate_distance(right_shoulder_x, right_shoulder_y, right_wrist_x, right_wrist_y)

    ax4.plot(time, left_arm_ext, 'b-', label='Left Fencer', linewidth=2)
    ax4.plot(time, right_arm_ext, 'r-', label='Right Fencer', linewidth=2)
    ax4.set_xlabel('Time (seconds)', fontsize=12)
    ax4.set_ylabel('Distance (pixels)', fontsize=12)
    ax4.set_title('Arm Extension (Shoulder-Wrist Distance)', fontsize=14, fontweight='bold')
    ax4.legend(fontsize=10)
    ax4.grid(True, alpha=0.3)

    # === 5. Distance Between Fencers ===
    ax5 = fig.add_subplot(gs[2, 1])

    # Use hip positions as center of mass approximation
    left_hip_y = left_y_px[str(KEYPOINTS['left_hip'])].values
    right_hip_y = right_y_px[str(KEYPOINTS['right_hip'])].values

    distance_between = calculate_distance(left_hip_x, left_hip_y, right_hip_x, right_hip_y)

    ax5.plot(time, distance_between, 'purple', linewidth=2)
    ax5.fill_between(time, distance_between, alpha=0.3, color='purple')
    ax5.set_xlabel('Time (seconds)', fontsize=12)
    ax5.set_ylabel('Distance (pixels)', fontsize=12)
    ax5.set_title('Distance Between Fencers (Hip-to-Hip)', fontsize=14, fontweight='bold')
    ax5.grid(True, alpha=0.3)

    # === 6. Movement Speed (Combined XY Velocity) ===
    ax6 = fig.add_subplot(gs[3, 0])

    left_hip_y_vals = left_y_px[str(KEYPOINTS['left_hip'])].values
    right_hip_y_vals = right_y_px[str(KEYPOINTS['right_hip'])].values

    left_vx = np.gradient(left_hip_x) * fps
    left_vy = np.gradient(left_hip_y_vals) * fps
    left_speed = np.sqrt(left_vx**2 + left_vy**2)

    right_vx = np.gradient(right_hip_x) * fps
    right_vy = np.gradient(right_hip_y_vals) * fps
    right_speed = np.sqrt(right_vx**2 + right_vy**2)

    left_speed_smooth = smooth_data(left_speed, window_length=5)
    right_speed_smooth = smooth_data(right_speed, window_length=5)

    ax6.plot(time, left_speed_smooth, 'b-', label='Left Fencer', linewidth=2)
    ax6.plot(time, right_speed_smooth, 'r-', label='Right Fencer', linewidth=2)
    ax6.set_xlabel('Time (seconds)', fontsize=12)
    ax6.set_ylabel('Speed (pixels/sec)', fontsize=12)
    ax6.set_title('Overall Movement Speed', fontsize=14, fontweight='bold')
    ax6.legend(fontsize=10)
    ax6.grid(True, alpha=0.3)

    # === 7. Pause Detection (Low Speed Periods) ===
    ax7 = fig.add_subplot(gs[3, 1])

    speed_threshold = 50  # pixels/sec
    left_pauses = left_speed_smooth < speed_threshold
    right_pauses = right_speed_smooth < speed_threshold

    ax7.fill_between(time, 0, 1, where=left_pauses, alpha=0.5, color='blue', label='Left Fencer Paused', step='mid')
    ax7.fill_between(time, 0, 0.5, where=right_pauses, alpha=0.5, color='red', label='Right Fencer Paused', step='mid')
    ax7.set_xlabel('Time (seconds)', fontsize=12)
    ax7.set_ylabel('Pause Indicator', fontsize=12)
    ax7.set_title(f'Movement Pauses (Speed < {speed_threshold} px/s)', fontsize=14, fontweight='bold')
    ax7.set_ylim([0, 1])
    ax7.legend(fontsize=10)
    ax7.grid(True, alpha=0.3)

    # === 8. Leg Extension (Hip to Ankle Distance) ===
    ax8 = fig.add_subplot(gs[4, 0])

    # Left fencer - right leg
    left_hip_r_x = left_x_px[str(KEYPOINTS['right_hip'])].values
    left_hip_r_y = left_y_px[str(KEYPOINTS['right_hip'])].values
    left_ankle_r_x = left_x_px[str(KEYPOINTS['right_ankle'])].values
    left_ankle_r_y = left_y_px[str(KEYPOINTS['right_ankle'])].values
    left_leg_ext = calculate_distance(left_hip_r_x, left_hip_r_y, left_ankle_r_x, left_ankle_r_y)

    # Right fencer - right leg
    right_hip_r_x = right_x_px[str(KEYPOINTS['right_hip'])].values
    right_hip_r_y = right_y_px[str(KEYPOINTS['right_hip'])].values
    right_ankle_r_x = right_x_px[str(KEYPOINTS['right_ankle'])].values
    right_ankle_r_y = right_y_px[str(KEYPOINTS['right_ankle'])].values
    right_leg_ext = calculate_distance(right_hip_r_x, right_hip_r_y, right_ankle_r_x, right_ankle_r_y)

    ax8.plot(time, left_leg_ext, 'b-', label='Left Fencer', linewidth=2)
    ax8.plot(time, right_leg_ext, 'r-', label='Right Fencer', linewidth=2)
    ax8.set_xlabel('Time (seconds)', fontsize=12)
    ax8.set_ylabel('Distance (pixels)', fontsize=12)
    ax8.set_title('Leg Extension (Hip-Ankle Distance)', fontsize=14, fontweight='bold')
    ax8.legend(fontsize=10)
    ax8.grid(True, alpha=0.3)

    # === 9. Body Height (Hip to Shoulder Distance - Vertical Posture) ===
    ax9 = fig.add_subplot(gs[4, 1])

    # Left fencer
    left_shoulder_avg_x = (left_x_px[str(KEYPOINTS['left_shoulder'])].values +
                           left_x_px[str(KEYPOINTS['right_shoulder'])].values) / 2
    left_shoulder_avg_y = (left_y_px[str(KEYPOINTS['left_shoulder'])].values +
                           left_y_px[str(KEYPOINTS['right_shoulder'])].values) / 2
    left_hip_avg_x = (left_x_px[str(KEYPOINTS['left_hip'])].values +
                      left_x_px[str(KEYPOINTS['right_hip'])].values) / 2
    left_hip_avg_y = (left_y_px[str(KEYPOINTS['left_hip'])].values +
                      left_y_px[str(KEYPOINTS['right_hip'])].values) / 2
    left_torso_length = calculate_distance(left_shoulder_avg_x, left_shoulder_avg_y,
                                           left_hip_avg_x, left_hip_avg_y)

    # Right fencer
    right_shoulder_avg_x = (right_x_px[str(KEYPOINTS['left_shoulder'])].values +
                            right_x_px[str(KEYPOINTS['right_shoulder'])].values) / 2
    right_shoulder_avg_y = (right_y_px[str(KEYPOINTS['left_shoulder'])].values +
                            right_y_px[str(KEYPOINTS['right_shoulder'])].values) / 2
    right_hip_avg_x = (right_x_px[str(KEYPOINTS['left_hip'])].values +
                       right_x_px[str(KEYPOINTS['right_hip'])].values) / 2
    right_hip_avg_y = (right_y_px[str(KEYPOINTS['left_hip'])].values +
                       right_y_px[str(KEYPOINTS['right_hip'])].values) / 2
    right_torso_length = calculate_distance(right_shoulder_avg_x, right_shoulder_avg_y,
                                            right_hip_avg_x, right_hip_avg_y)

    ax9.plot(time, left_torso_length, 'b-', label='Left Fencer', linewidth=2)
    ax9.plot(time, right_torso_length, 'r-', label='Right Fencer', linewidth=2)
    ax9.set_xlabel('Time (seconds)', fontsize=12)
    ax9.set_ylabel('Distance (pixels)', fontsize=12)
    ax9.set_title('Torso Length (Shoulder-Hip Distance)', fontsize=14, fontweight='bold')
    ax9.legend(fontsize=10)
    ax9.grid(True, alpha=0.3)

    # === 10. 2D Movement Trajectory ===
    ax10 = fig.add_subplot(gs[5, :])

    # Plot trajectories
    ax10.plot(left_hip_x, left_hip_y_vals, 'b-', label='Left Fencer', linewidth=2, alpha=0.7)
    ax10.plot(right_hip_x, right_hip_y_vals, 'r-', label='Right Fencer', linewidth=2, alpha=0.7)

    # Mark start and end positions
    ax10.scatter(left_hip_x[0], left_hip_y_vals[0], c='blue', s=200, marker='o',
                edgecolors='black', linewidths=2, label='Left Start', zorder=5)
    ax10.scatter(left_hip_x[-1], left_hip_y_vals[-1], c='blue', s=200, marker='s',
                edgecolors='black', linewidths=2, label='Left End', zorder=5)
    ax10.scatter(right_hip_x[0], right_hip_y_vals[0], c='red', s=200, marker='o',
                edgecolors='black', linewidths=2, label='Right Start', zorder=5)
    ax10.scatter(right_hip_x[-1], right_hip_y_vals[-1], c='red', s=200, marker='s',
                edgecolors='black', linewidths=2, label='Right End', zorder=5)

    ax10.set_xlabel('X Position (pixels)', fontsize=12)
    ax10.set_ylabel('Y Position (pixels)', fontsize=12)
    ax10.set_title('2D Movement Trajectory (Hip Position)', fontsize=14, fontweight='bold')
    ax10.legend(fontsize=10, ncol=2)
    ax10.grid(True, alpha=0.3)
    ax10.axis('equal')

    # Add main title
    fig.suptitle(f'Fencing Movement Analysis - {os.path.basename(excel_file)}',
                fontsize=16, fontweight='bold', y=0.995)

    # Save figure
    output_file = f"{output_prefix}_analysis.png"
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"✓ Saved comprehensive analysis: {output_file}")
    plt.close()

    # Generate summary statistics
    print(f"\n  Summary Statistics:")
    print(f"  - Frames: {num_frames}")
    print(f"  - Duration: {num_frames/fps:.2f} seconds")
    print(f"  - Left Fencer:")
    print(f"    • Avg Speed: {np.mean(left_speed_smooth):.2f} px/s")
    print(f"    • Max Speed: {np.max(left_speed_smooth):.2f} px/s")
    print(f"    • Avg Arm Extension: {np.mean(left_arm_ext):.2f} px")
    print(f"    • Pause Time: {np.sum(left_pauses)/fps:.2f}s ({100*np.sum(left_pauses)/num_frames:.1f}%)")
    print(f"  - Right Fencer:")
    print(f"    • Avg Speed: {np.mean(right_speed_smooth):.2f} px/s")
    print(f"    • Max Speed: {np.max(right_speed_smooth):.2f} px/s")
    print(f"    • Avg Arm Extension: {np.mean(right_arm_ext):.2f} px")
    print(f"    • Pause Time: {np.sum(right_pauses)/fps:.2f}s ({100*np.sum(right_pauses)/num_frames:.1f}%)")
    print(f"  - Engagement:")
    print(f"    • Min Distance: {np.min(distance_between):.2f} px")
    print(f"    • Max Distance: {np.max(distance_between):.2f} px")
    print(f"    • Avg Distance: {np.mean(distance_between):.2f} px")

if __name__ == "__main__":
    # Analyze both videos
    analyze_video("/workspace/Project/run_sam_keypoints/3809_keypoints.xlsx",
                  "/workspace/Project/run_sam_keypoints/3809", fps=30)

    analyze_video("/workspace/Project/run_sam_keypoints/3810_keypoints.xlsx",
                  "/workspace/Project/run_sam_keypoints/3810", fps=30)

    print("\n✓ All analyses complete!")
