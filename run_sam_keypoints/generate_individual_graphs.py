#!/usr/bin/env python3
"""
Generate individual analysis graphs for fencing videos.
Creates separate image files for each metric.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
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

def calculate_distance(x1, y1, x2, y2):
    """Calculate Euclidean distance between two points."""
    return np.sqrt((x2 - x1)**2 + (y2 - y1)**2)

def smooth_data(data, window_length=5, polyorder=2):
    """Smooth data using Savitzky-Golay filter."""
    if len(data) < window_length:
        return data
    return savgol_filter(data, window_length, polyorder)

def save_graph(fig, output_dir, filename):
    """Save a graph to file."""
    filepath = os.path.join(output_dir, filename)
    fig.savefig(filepath, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  ✓ {filename}")

def analyze_video_individual(excel_file, output_dir, fps=30, scale_factor=126.06):
    """Generate individual analysis graphs for a video."""

    print(f"\nGenerating individual graphs for {os.path.basename(excel_file)}...")

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Load data
    left_x, left_y, right_x, right_y = load_data(excel_file)
    num_frames = len(left_x)
    time = np.arange(num_frames) / fps

    # Convert to pixel coordinates
    left_x_px = left_x * scale_factor
    left_y_px = left_y * scale_factor
    right_x_px = right_x * scale_factor
    right_y_px = right_y * scale_factor

    # Calculate common metrics
    left_hip_x = left_x_px[str(KEYPOINTS['left_hip'])].values
    left_hip_y = left_y_px[str(KEYPOINTS['left_hip'])].values
    right_hip_x = right_x_px[str(KEYPOINTS['right_hip'])].values
    right_hip_y = right_y_px[str(KEYPOINTS['right_hip'])].values

    # === 1. X-Coordinate Movement ===
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(time, left_hip_x, 'b-', label='Left Fencer Hip', linewidth=2.5)
    ax.plot(time, right_hip_x, 'r-', label='Right Fencer Hip', linewidth=2.5)
    ax.set_xlabel('Time (seconds)', fontsize=14)
    ax.set_ylabel('X Position (pixels)', fontsize=14)
    ax.set_title('Horizontal Movement Over Time (Hip Position)', fontsize=16, fontweight='bold')
    ax.legend(fontsize=12)
    ax.grid(True, alpha=0.3)
    save_graph(fig, output_dir, '01_horizontal_movement.png')

    # === 2. Velocity ===
    left_velocity = np.gradient(left_hip_x) * fps
    right_velocity = np.gradient(right_hip_x) * fps
    left_velocity_smooth = smooth_data(left_velocity, window_length=5)
    right_velocity_smooth = smooth_data(right_velocity, window_length=5)

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(time, left_velocity_smooth, 'b-', label='Left Fencer', linewidth=2.5)
    ax.plot(time, right_velocity_smooth, 'r-', label='Right Fencer', linewidth=2.5)
    ax.axhline(y=0, color='k', linestyle='--', alpha=0.3)
    ax.set_xlabel('Time (seconds)', fontsize=14)
    ax.set_ylabel('Velocity (pixels/sec)', fontsize=14)
    ax.set_title('Horizontal Velocity', fontsize=16, fontweight='bold')
    ax.legend(fontsize=12)
    ax.grid(True, alpha=0.3)
    save_graph(fig, output_dir, '02_velocity.png')

    # === 3. Acceleration ===
    left_accel = np.gradient(left_velocity_smooth) * fps
    right_accel = np.gradient(right_velocity_smooth) * fps

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(time, left_accel, 'b-', label='Left Fencer', linewidth=2.5)
    ax.plot(time, right_accel, 'r-', label='Right Fencer', linewidth=2.5)
    ax.axhline(y=0, color='k', linestyle='--', alpha=0.3)
    ax.set_xlabel('Time (seconds)', fontsize=14)
    ax.set_ylabel('Acceleration (pixels/sec²)', fontsize=14)
    ax.set_title('Horizontal Acceleration', fontsize=16, fontweight='bold')
    ax.legend(fontsize=12)
    ax.grid(True, alpha=0.3)
    save_graph(fig, output_dir, '03_acceleration.png')

    # === 4. Arm Extension ===
    # Left fencer - right arm
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

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(time, left_arm_ext, 'b-', label='Left Fencer', linewidth=2.5)
    ax.plot(time, right_arm_ext, 'r-', label='Right Fencer', linewidth=2.5)
    ax.set_xlabel('Time (seconds)', fontsize=14)
    ax.set_ylabel('Distance (pixels)', fontsize=14)
    ax.set_title('Arm Extension (Shoulder-Wrist Distance)', fontsize=16, fontweight='bold')
    ax.legend(fontsize=12)
    ax.grid(True, alpha=0.3)
    save_graph(fig, output_dir, '04_arm_extension.png')

    # === 5. Distance Between Fencers ===
    distance_between = calculate_distance(left_hip_x, left_hip_y, right_hip_x, right_hip_y)

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(time, distance_between, 'purple', linewidth=2.5)
    ax.fill_between(time, distance_between, alpha=0.3, color='purple')
    ax.set_xlabel('Time (seconds)', fontsize=14)
    ax.set_ylabel('Distance (pixels)', fontsize=14)
    ax.set_title('Distance Between Fencers (Hip-to-Hip)', fontsize=16, fontweight='bold')
    ax.grid(True, alpha=0.3)
    save_graph(fig, output_dir, '05_distance_between_fencers.png')

    # === 6. Overall Movement Speed ===
    left_vx = np.gradient(left_hip_x) * fps
    left_vy = np.gradient(left_hip_y) * fps
    left_speed = np.sqrt(left_vx**2 + left_vy**2)

    right_vx = np.gradient(right_hip_x) * fps
    right_vy = np.gradient(right_hip_y) * fps
    right_speed = np.sqrt(right_vx**2 + right_vy**2)

    left_speed_smooth = smooth_data(left_speed, window_length=5)
    right_speed_smooth = smooth_data(right_speed, window_length=5)

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(time, left_speed_smooth, 'b-', label='Left Fencer', linewidth=2.5)
    ax.plot(time, right_speed_smooth, 'r-', label='Right Fencer', linewidth=2.5)
    ax.set_xlabel('Time (seconds)', fontsize=14)
    ax.set_ylabel('Speed (pixels/sec)', fontsize=14)
    ax.set_title('Overall Movement Speed', fontsize=16, fontweight='bold')
    ax.legend(fontsize=12)
    ax.grid(True, alpha=0.3)
    save_graph(fig, output_dir, '06_movement_speed.png')

    # === 7. Pause Detection ===
    speed_threshold = 50
    left_pauses = left_speed_smooth < speed_threshold
    right_pauses = right_speed_smooth < speed_threshold

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.fill_between(time, 0, 1, where=left_pauses, alpha=0.5, color='blue', label='Left Fencer Paused', step='mid')
    ax.fill_between(time, 0, 0.5, where=right_pauses, alpha=0.5, color='red', label='Right Fencer Paused', step='mid')
    ax.set_xlabel('Time (seconds)', fontsize=14)
    ax.set_ylabel('Pause Indicator', fontsize=14)
    ax.set_title(f'Movement Pauses (Speed < {speed_threshold} px/s)', fontsize=16, fontweight='bold')
    ax.set_ylim([0, 1])
    ax.legend(fontsize=12)
    ax.grid(True, alpha=0.3)
    save_graph(fig, output_dir, '07_movement_pauses.png')

    # === 8. Leg Extension ===
    left_hip_r_x = left_x_px[str(KEYPOINTS['right_hip'])].values
    left_hip_r_y = left_y_px[str(KEYPOINTS['right_hip'])].values
    left_ankle_r_x = left_x_px[str(KEYPOINTS['right_ankle'])].values
    left_ankle_r_y = left_y_px[str(KEYPOINTS['right_ankle'])].values
    left_leg_ext = calculate_distance(left_hip_r_x, left_hip_r_y, left_ankle_r_x, left_ankle_r_y)

    right_hip_r_x = right_x_px[str(KEYPOINTS['right_hip'])].values
    right_hip_r_y = right_y_px[str(KEYPOINTS['right_hip'])].values
    right_ankle_r_x = right_x_px[str(KEYPOINTS['right_ankle'])].values
    right_ankle_r_y = right_y_px[str(KEYPOINTS['right_ankle'])].values
    right_leg_ext = calculate_distance(right_hip_r_x, right_hip_r_y, right_ankle_r_x, right_ankle_r_y)

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(time, left_leg_ext, 'b-', label='Left Fencer', linewidth=2.5)
    ax.plot(time, right_leg_ext, 'r-', label='Right Fencer', linewidth=2.5)
    ax.set_xlabel('Time (seconds)', fontsize=14)
    ax.set_ylabel('Distance (pixels)', fontsize=14)
    ax.set_title('Leg Extension (Hip-Ankle Distance)', fontsize=16, fontweight='bold')
    ax.legend(fontsize=12)
    ax.grid(True, alpha=0.3)
    save_graph(fig, output_dir, '08_leg_extension.png')

    # === 9. Torso Length ===
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

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(time, left_torso_length, 'b-', label='Left Fencer', linewidth=2.5)
    ax.plot(time, right_torso_length, 'r-', label='Right Fencer', linewidth=2.5)
    ax.set_xlabel('Time (seconds)', fontsize=14)
    ax.set_ylabel('Distance (pixels)', fontsize=14)
    ax.set_title('Torso Length (Shoulder-Hip Distance)', fontsize=16, fontweight='bold')
    ax.legend(fontsize=12)
    ax.grid(True, alpha=0.3)
    save_graph(fig, output_dir, '09_torso_length.png')

    # === 10. 2D Trajectory ===
    fig, ax = plt.subplots(figsize=(12, 10))
    ax.plot(left_hip_x, left_hip_y, 'b-', label='Left Fencer', linewidth=2.5, alpha=0.7)
    ax.plot(right_hip_x, right_hip_y, 'r-', label='Right Fencer', linewidth=2.5, alpha=0.7)

    ax.scatter(left_hip_x[0], left_hip_y[0], c='blue', s=250, marker='o',
              edgecolors='black', linewidths=2, label='Left Start', zorder=5)
    ax.scatter(left_hip_x[-1], left_hip_y[-1], c='blue', s=250, marker='s',
              edgecolors='black', linewidths=2, label='Left End', zorder=5)
    ax.scatter(right_hip_x[0], right_hip_y[0], c='red', s=250, marker='o',
              edgecolors='black', linewidths=2, label='Right Start', zorder=5)
    ax.scatter(right_hip_x[-1], right_hip_y[-1], c='red', s=250, marker='s',
              edgecolors='black', linewidths=2, label='Right End', zorder=5)

    ax.set_xlabel('X Position (pixels)', fontsize=14)
    ax.set_ylabel('Y Position (pixels)', fontsize=14)
    ax.set_title('2D Movement Trajectory (Hip Position)', fontsize=16, fontweight='bold')
    ax.legend(fontsize=12, ncol=2)
    ax.grid(True, alpha=0.3)
    ax.axis('equal')
    save_graph(fig, output_dir, '10_trajectory_2d.png')

    print(f"✓ Generated 10 individual graphs in: {output_dir}")

if __name__ == "__main__":
    # Analyze both videos
    analyze_video_individual(
        "/workspace/Project/run_sam_keypoints/3809_keypoints.xlsx",
        "/workspace/Project/run_sam_keypoints/3809_analysis_graphs",
        fps=30
    )

    analyze_video_individual(
        "/workspace/Project/run_sam_keypoints/3810_keypoints.xlsx",
        "/workspace/Project/run_sam_keypoints/3810_analysis_graphs",
        fps=30
    )

    print("\n✓ All individual graphs generated!")
