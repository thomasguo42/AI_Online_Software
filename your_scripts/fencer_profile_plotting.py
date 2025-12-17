#!/usr/bin/env python3
"""
Fencer Profile Plotting Module

This module creates comprehensive visualizations for individual fencer profiles,
adapting the existing bout analysis graphs for fencer-specific insights.
Generates radar charts, performance bar charts, attack analysis, defense analysis, etc.
"""

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import os
import logging
from typing import Dict, List, Any, Optional

# Set visualization style
plt.style.use('default')
sns.set_palette("husl")
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

def create_fencer_profile_radar_chart(fencer_data: Dict[str, Any], fencer_name: str, ax: plt.Axes) -> None:
    """
    Create a radar chart showing overall fencer performance metrics.
    
    Args:
        fencer_data: Dictionary containing fencer metrics and bout data
        fencer_name: Name of the fencer (e.g., 'Fencer_Left')
        ax: Matplotlib axes to plot on
    """
    metrics = fencer_data.get('metrics', {})
    
    # Define radar chart metrics with normalized values (0-10 scale)
    categories = [
        'Reaction Speed',
        'Velocity',
        'Acceleration', 
        'Forward Pressure',
        'Attack Frequency',
        'Extension Speed',
        'Launch Efficiency',
        'Distance Control'
    ]
    
    # Normalize metrics to 0-10 scale
    def normalize_metric(value, min_val, max_val):
        if max_val == min_val:
            return 5.0  # neutral value
        return max(0, min(10, ((value - min_val) / (max_val - min_val)) * 10))
    
    # Calculate normalized values
    values = [
        10 - normalize_metric(metrics.get('avg_first_step_init', 0.15), 0.05, 0.3),  # Reaction speed (inverted - lower is better)
        normalize_metric(metrics.get('avg_velocity', 1.0), 0.5, 3.0),  # Velocity
        normalize_metric(metrics.get('avg_acceleration', 5.0), 2.0, 30.0),  # Acceleration
        normalize_metric(metrics.get('avg_advance_ratio', 0.5), 0.2, 1.0),  # Forward pressure
        normalize_metric(metrics.get('attacking_ratio', 0.0), 0.0, 1.0),  # Attack frequency
        normalize_metric(metrics.get('avg_arm_extension_duration', 0.5), 0.2, 1.0),  # Extension speed (inverted)
        normalize_metric(metrics.get('avg_launch_promptness', 0.5), 0.1, 1.0),  # Launch efficiency
        normalize_metric(1 - metrics.get('avg_pause_ratio', 0.5), 0.2, 0.8)  # Distance control
    ]
    
    # Number of variables
    N = len(categories)
    
    # Compute angle for each axis
    angles = [n / float(N) * 2 * np.pi for n in range(N)]
    angles += angles[:1]  # Complete the circle
    
    # Add first value to end to close the radar chart
    values += values[:1]
    
    # Plot
    ax.plot(angles, values, 'o-', linewidth=2, label=fencer_name, color='#2196F3')
    ax.fill(angles, values, alpha=0.25, color='#2196F3')
    
    # Add labels
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories, fontsize=10)
    ax.set_ylim(0, 10)
    ax.set_yticks([2, 4, 6, 8, 10])
    ax.set_yticklabels(['2', '4', '6', '8', '10'], fontsize=8)
    ax.grid(True, alpha=0.3)
    
    # Add title
    ax.set_title(f'{fencer_name} - Performance Profile', size=14, fontweight='bold', pad=20)

def create_attack_type_analysis(fencer_data: Dict[str, Any], fencer_name: str, ax: plt.Axes) -> None:
    """
    Create attack type frequency and success analysis.
    """
    bouts = fencer_data.get('bouts', [])
    
    # Aggregate attack data across all bouts
    attack_counts = {}
    attack_victories = {}
    
    for bout in bouts:
        interval_analysis = bout.get('metrics', {}).get('interval_analysis', {})
        summary = interval_analysis.get('summary', {})
        attacks = summary.get('attacks', {})
        
        # Determine if this bout was won (you may need to adjust this logic based on your data structure)
        is_winner = bout.get('is_winner', False)  # This might need adjustment based on actual data
        
        # Count each attack type
        for attack_type in ['simple', 'compound', 'holding', 'preparations']:
            count = attacks.get(attack_type, 0)
            if count > 0:
                if attack_type not in attack_counts:
                    attack_counts[attack_type] = 0
                    attack_victories[attack_type] = 0
                attack_counts[attack_type] += count
                if is_winner:
                    attack_victories[attack_type] += count
    
    if not attack_counts:
        ax.text(0.5, 0.5, 'No attack data available', ha='center', va='center', 
                transform=ax.transAxes, fontsize=12)
        ax.set_title(f'{fencer_name} - Attack Type Analysis')
        return
    
    # Prepare data for plotting
    attack_types = list(attack_counts.keys())
    counts = list(attack_counts.values())
    success_rates = [(attack_victories.get(at, 0) / attack_counts[at] * 100) 
                    if attack_counts[at] > 0 else 0 for at in attack_types]
    
    # Create grouped bar chart
    x = np.arange(len(attack_types))
    width = 0.35
    
    bars1 = ax.bar(x - width/2, counts, width, label='Total Uses', alpha=0.8, color='#2196F3')
    bars2 = ax.bar(x + width/2, success_rates, width, label='Success Rate %', alpha=0.8, color='#4CAF50')
    
    # Add value labels
    for bar in bars1:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.1,
               f'{int(height)}', ha='center', va='bottom', fontweight='bold')
    
    for bar in bars2:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 1,
               f'{height:.1f}%', ha='center', va='bottom', fontweight='bold')
    
    ax.set_xlabel('Attack Type')
    ax.set_ylabel('Count / Success Rate %')
    ax.set_title(f'{fencer_name} - Attack Type Analysis')
    ax.set_xticks(x)
    ax.set_xticklabels([at.replace('_', ' ').title() for at in attack_types], rotation=45, ha='right')
    ax.legend()
    ax.grid(True, alpha=0.3)

def create_tempo_analysis(fencer_data: Dict[str, Any], fencer_name: str, ax: plt.Axes) -> None:
    """
    Create tempo type frequency analysis.
    """
    bouts = fencer_data.get('bouts', [])
    
    # Aggregate tempo data
    tempo_counts = {}
    
    for bout in bouts:
        interval_analysis = bout.get('metrics', {}).get('interval_analysis', {})
        summary = interval_analysis.get('summary', {})
        tempo = summary.get('tempo', {})
        
        for tempo_type in ['steady', 'variable', 'broken']:
            count = tempo.get(tempo_type, 0)
            if count > 0:
                if tempo_type not in tempo_counts:
                    tempo_counts[tempo_type] = 0
                tempo_counts[tempo_type] += count
    
    if not tempo_counts:
        ax.text(0.5, 0.5, 'No tempo data available', ha='center', va='center', 
                transform=ax.transAxes, fontsize=12)
        ax.set_title(f'{fencer_name} - Tempo Analysis')
        return
    
    # Create pie chart
    labels = [f"{tempo.replace('_', ' ').title()}\n({count})" 
             for tempo, count in tempo_counts.items()]
    sizes = list(tempo_counts.values())
    colors = ['#4CAF50', '#FF9800', '#FF5722'][:len(sizes)]
    
    wedges, texts, autotexts = ax.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%',
                                     shadow=True, startangle=90)
    
    # Enhance text
    for autotext in autotexts:
        autotext.set_color('white')
        autotext.set_fontweight('bold')
    
    ax.set_title(f'{fencer_name} - Tempo Distribution')

def create_performance_metrics_bar_chart(fencer_data: Dict[str, Any], fencer_name: str, ax: plt.Axes) -> None:
    """
    Create bar chart of key performance metrics.
    """
    metrics = fencer_data.get('metrics', {})
    
    # Define metrics to display
    metric_names = [
        'Avg Velocity\n(m/s)',
        'Avg Acceleration\n(m/s²)',
        'Forward Pressure\n(%)',
        'Reaction Time\n(s)',
        'Extension Freq.\n(per bout)',
        'Launch Success\n(%)'
    ]
    
    metric_values = [
        metrics.get('avg_velocity', 0),
        metrics.get('avg_acceleration', 0) / 10,  # Scale down for visibility
        metrics.get('avg_advance_ratio', 0) * 100,
        metrics.get('avg_first_step_init', 0),
        metrics.get('total_arm_extensions', 0) / max(len(fencer_data.get('bouts', [])), 1),
        metrics.get('attacking_ratio', 0) * 100
    ]
    
    # Create color map based on performance (green for good, red for poor)
    colors = []
    for i, value in enumerate(metric_values):
        if i in [0, 1, 2, 4, 5]:  # Higher is better
            colors.append('#4CAF50' if value > np.mean(metric_values) else '#FF5722')
        else:  # Lower is better (reaction time)
            colors.append('#4CAF50' if value < 0.15 else '#FF5722')
    
    bars = ax.bar(range(len(metric_names)), metric_values, color=colors, alpha=0.8)
    
    # Add value labels
    for bar, value in zip(bars, metric_values):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + max(metric_values) * 0.01,
               f'{value:.2f}', ha='center', va='bottom', fontweight='bold')
    
    ax.set_xlabel('Performance Metrics')
    ax.set_ylabel('Metric Value')
    ax.set_title(f'{fencer_name} - Key Performance Indicators')
    ax.set_xticks(range(len(metric_names)))
    ax.set_xticklabels(metric_names, rotation=45, ha='right')
    ax.grid(True, alpha=0.3)

def create_distance_management_analysis(fencer_data: Dict[str, Any], fencer_name: str, ax: plt.Axes) -> None:
    """
    Create distance management effectiveness analysis.
    """
    bouts = fencer_data.get('bouts', [])
    
    # Aggregate distance data
    good_attack_distances = 0
    poor_attack_distances = 0
    good_defensive_distances = 0
    poor_defensive_distances = 0
    
    for bout in bouts:
        interval_analysis = bout.get('metrics', {}).get('interval_analysis', {})
        summary = interval_analysis.get('summary', {})
        distance = summary.get('distance', {})
        defense = summary.get('defense', {})
        
        good_attack_distances += distance.get('good_attack_distances', 0)
        poor_attack_distances += distance.get('missed_opportunities', 0)
        good_defensive_distances += defense.get('good_distance_management', 0)
        poor_defensive_distances += defense.get('counter_opportunities', 0) - defense.get('counters_executed', 0)
    
    # Prepare data
    categories = ['Attack Distance\nManagement', 'Defensive Distance\nManagement']
    good_counts = [good_attack_distances, good_defensive_distances]
    poor_counts = [poor_attack_distances, poor_defensive_distances]
    
    # Create grouped bar chart
    x = np.arange(len(categories))
    width = 0.35
    
    bars1 = ax.bar(x - width/2, good_counts, width, label='Effective', alpha=0.8, color='#4CAF50')
    bars2 = ax.bar(x + width/2, poor_counts, width, label='Ineffective', alpha=0.8, color='#FF5722')
    
    # Add value labels
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            if height > 0:
                ax.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                       f'{int(height)}', ha='center', va='bottom', fontweight='bold')
    
    ax.set_xlabel('Distance Management Type')
    ax.set_ylabel('Frequency')
    ax.set_title(f'{fencer_name} - Distance Management Analysis')
    ax.set_xticks(x)
    ax.set_xticklabels(categories)
    ax.legend()
    ax.grid(True, alpha=0.3)

def create_bout_progression_chart(fencer_data: Dict[str, Any], fencer_name: str, ax: plt.Axes) -> None:
    """
    Create line chart showing performance progression across bouts.
    """
    bouts = fencer_data.get('bouts', [])
    
    if len(bouts) < 2:
        ax.text(0.5, 0.5, 'Insufficient bout data for progression analysis', 
                ha='center', va='center', transform=ax.transAxes, fontsize=12)
        ax.set_title(f'{fencer_name} - Bout Progression')
        return
    
    # Extract progression metrics
    bout_numbers = [bout.get('match_idx', i+1) for i, bout in enumerate(bouts)]
    velocities = [bout.get('metrics', {}).get('velocity', 0) for bout in bouts]
    attack_scores = [bout.get('metrics', {}).get('attacking_score', 0) for bout in bouts]
    advance_ratios = [bout.get('metrics', {}).get('advance_ratio', 0) * 100 for bout in bouts]
    
    # Create multiple line plots
    ax2 = ax.twinx()  # Second y-axis for different scale metrics
    
    line1 = ax.plot(bout_numbers, velocities, 'o-', label='Velocity (m/s)', color='#2196F3', linewidth=2)
    line2 = ax.plot(bout_numbers, advance_ratios, 's-', label='Forward Pressure (%)', color='#4CAF50', linewidth=2)
    line3 = ax2.plot(bout_numbers, attack_scores, '^-', label='Attack Score', color='#FF9800', linewidth=2)
    
    # Formatting
    ax.set_xlabel('Bout Number')
    ax.set_ylabel('Velocity (m/s) / Forward Pressure (%)')
    ax2.set_ylabel('Attack Score')
    ax.set_title(f'{fencer_name} - Performance Progression')
    
    # Combine legends
    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2, loc='upper left')
    
    ax.grid(True, alpha=0.3)

def save_fencer_profile_plots(fencer_data: Dict[str, Any], fencer_name: str, output_dir: str) -> Dict[str, str]:
    """
    Generate and save all fencer profile visualization plots.
    
    Args:
        fencer_data: Dictionary containing fencer metrics and bout data
        fencer_name: Name of the fencer (e.g., 'Fencer_Left')
        output_dir: Directory to save plots
        
    Returns:
        Dictionary with plot file paths
    """
    os.makedirs(output_dir, exist_ok=True)
    plot_files = {}
    
    # Create comprehensive figure with all plots
    fig = plt.figure(figsize=(20, 24))
    
    # Plot 1: Radar Chart (Performance Profile)
    ax1 = plt.subplot(4, 2, 1, projection='polar')
    create_fencer_profile_radar_chart(fencer_data, fencer_name, ax1)
    
    # Plot 2: Attack Type Analysis
    ax2 = plt.subplot(4, 2, 2)
    create_attack_type_analysis(fencer_data, fencer_name, ax2)
    
    # Plot 3: Tempo Analysis
    ax3 = plt.subplot(4, 2, 3)
    create_tempo_analysis(fencer_data, fencer_name, ax3)
    
    # Plot 4: Performance Metrics Bar Chart
    ax4 = plt.subplot(4, 2, 4)
    create_performance_metrics_bar_chart(fencer_data, fencer_name, ax4)
    
    # Plot 5: Distance Management Analysis
    ax5 = plt.subplot(4, 2, 5)
    create_distance_management_analysis(fencer_data, fencer_name, ax5)
    
    # Plot 6: Bout Progression
    ax6 = plt.subplot(4, 2, 6)
    create_bout_progression_chart(fencer_data, fencer_name, ax6)
    
    # Plot 7: Tag Analysis (based on our enhanced tagging system)
    ax7 = plt.subplot(4, 2, 7)
    create_tag_analysis_chart(fencer_data, fencer_name, ax7)
    
    # Plot 8: Victory/Loss Analysis
    ax8 = plt.subplot(4, 2, 8)
    create_victory_analysis_chart(fencer_data, fencer_name, ax8)
    
    plt.tight_layout()
    
    # Save comprehensive plot
    comprehensive_path = os.path.join(output_dir, f'{fencer_name.lower()}_profile_analysis.png')
    plt.savefig(comprehensive_path, dpi=300, bbox_inches='tight')
    plot_files['comprehensive_profile'] = comprehensive_path
    
    plt.close()
    
    # Save individual radar chart for quick reference
    fig_radar, ax_radar = plt.subplots(figsize=(10, 10), subplot_kw=dict(projection='polar'))
    create_fencer_profile_radar_chart(fencer_data, fencer_name, ax_radar)
    plt.tight_layout()
    
    radar_path = os.path.join(output_dir, f'{fencer_name.lower()}_radar_profile.png')
    plt.savefig(radar_path, dpi=300, bbox_inches='tight')
    plot_files['radar_profile'] = radar_path
    
    plt.close()
    
    logging.info(f"Generated fencer profile plots for {fencer_name} in {output_dir}")
    return plot_files

def create_tag_analysis_chart(fencer_data: Dict[str, Any], fencer_name: str, ax: plt.Axes) -> None:
    """
    Create analysis chart based on fencer tags from enhanced tagging system.
    """
    # This would integrate with the tagging system we created earlier
    # For now, we'll create a placeholder that analyzes basic performance patterns
    
    bouts = fencer_data.get('bouts', [])
    
    # Simulate tag analysis based on available data
    positive_tags = []
    negative_tags = []
    
    metrics = fencer_data.get('metrics', {})
    
    # Analyze performance to assign tags
    if metrics.get('avg_velocity', 0) > 2.0:
        positive_tags.append('High Speed')
    elif metrics.get('avg_velocity', 0) < 1.0:
        negative_tags.append('Low Speed')
    
    if metrics.get('avg_first_step_init', 0.15) < 0.1:
        positive_tags.append('Fast Reaction')
    elif metrics.get('avg_first_step_init', 0.15) > 0.2:
        negative_tags.append('Slow Reaction')
        
    if metrics.get('avg_advance_ratio', 0.5) > 0.7:
        positive_tags.append('High Pressure')
    elif metrics.get('avg_advance_ratio', 0.5) < 0.3:
        negative_tags.append('Low Pressure')
    
    # Create stacked bar chart
    categories = ['Strengths', 'Weaknesses']
    positive_count = len(positive_tags)
    negative_count = len(negative_tags)
    
    if positive_count == 0 and negative_count == 0:
        ax.text(0.5, 0.5, 'Performance analysis in progress', ha='center', va='center', 
                transform=ax.transAxes, fontsize=12)
        ax.set_title(f'{fencer_name} - Performance Tags')
        return
    
    bars1 = ax.bar(categories, [positive_count, 0], color='#4CAF50', alpha=0.8, label='Positive Traits')
    bars2 = ax.bar(categories, [0, negative_count], color='#FF5722', alpha=0.8, label='Areas to Improve')
    
    # Add count labels
    if positive_count > 0:
        ax.text(0, positive_count/2, f'{positive_count}', ha='center', va='center', 
                color='white', fontweight='bold')
    if negative_count > 0:
        ax.text(1, negative_count/2, f'{negative_count}', ha='center', va='center', 
                color='white', fontweight='bold')
    
    ax.set_ylabel('Number of Traits')
    ax.set_title(f'{fencer_name} - Performance Profile')
    ax.legend()
    ax.grid(True, alpha=0.3)

def create_victory_analysis_chart(fencer_data: Dict[str, Any], fencer_name: str, ax: plt.Axes) -> None:
    """
    Create victory/loss pattern analysis.
    """
    # This would require bout result data - for now we'll create a placeholder
    ax.text(0.5, 0.5, 'Victory analysis requires\nbout result integration', 
            ha='center', va='center', transform=ax.transAxes, fontsize=12)
    ax.set_title(f'{fencer_name} - Victory Patterns')

def generate_fencer_profile_graphs(upload_id: int, base_output_dir: str) -> Dict[str, Dict[str, str]]:
    """
    Main function to generate fencer profile graphs for both fencers.
    
    Args:
        upload_id: Upload ID to process
        base_output_dir: Base directory for saving plots
        
    Returns:
        Dictionary with plot file paths for each fencer
    """
    try:
        # Load fencer analysis data
        fencer_dir = os.path.join(base_output_dir, str(upload_id), 'fencer_analysis')
        
        if not os.path.exists(fencer_dir):
            logging.error(f"Fencer analysis directory not found: {fencer_dir}")
            return {}
        
        plot_results = {}
        
        # Process each fencer
        for fencer_name in ['Fencer_Left', 'Fencer_Right']:
            fencer_file = os.path.join(fencer_dir, f'fencer_{fencer_name}_analysis.json')
            
            if os.path.exists(fencer_file):
                try:
                    with open(fencer_file, 'r', encoding='utf-8') as f:
                        fencer_data = json.load(f)
                    
                    # Create output directory for this fencer's plots
                    fencer_output_dir = os.path.join(fencer_dir, 'profile_plots', fencer_name)
                    
                    # Generate plots
                    plot_files = save_fencer_profile_plots(fencer_data, fencer_name, fencer_output_dir)
                    plot_results[fencer_name] = plot_files
                    
                    logging.info(f"Generated profile plots for {fencer_name}")
                    
                except Exception as e:
                    logging.error(f"Error processing {fencer_name}: {str(e)}")
            else:
                logging.warning(f"Fencer analysis file not found: {fencer_file}")
        
        return plot_results
    
    except Exception as e:
        logging.error(f"Error generating fencer profile graphs: {str(e)}")
        return {}

if __name__ == "__main__":
    # Test with a sample upload_id
    import argparse
    
    parser = argparse.ArgumentParser(description='Generate fencer profile graphs')
    parser.add_argument('--upload_id', type=int, required=True, help='Upload ID to process')
    parser.add_argument('--output_dir', type=str, default='/workspace/Project/results', 
                       help='Base output directory')
    
    args = parser.parse_args()
    
    results = generate_fencer_profile_graphs(args.upload_id, args.output_dir)
    print(f"Generated graphs for {len(results)} fencers")
    for fencer, files in results.items():
        print(f"{fencer}: {len(files)} plot files generated")