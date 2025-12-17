#!/usr/bin/env python3
"""
Plotting functions for fencing analysis visualizations.
Each function creates a specific chart type optimized for fencing tactical analysis.
"""

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import os
import logging
import json

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

# Translations
TITLE_TRANSLATIONS = {
    "Attack Type & Victory Analysis": "攻击类型与胜利分析",
    "Tempo Type & Victory Analysis": "节奏类型与胜利分析",
    "Attack Distance Analysis": "攻击距离分析",
    "Counter Opportunity Analysis": "反击机会分析",
    "Retreat Quality Analysis": "退却质量分析",
    "Retreat Distance Analysis": "退却距离分析",
    "Defensive Quality Analysis": "防守质量分析",
    "Bout Outcome Analysis": "回合结果分析",
    "Left Fencer": "左侧击剑手",
    "Right Fencer": "右侧击剑手"
}

LABEL_TRANSLATIONS = {
    "attack_type": "攻击类型",
    "tempo_type": "节奏类型",
    "bout_idx": "回合索引",
    "total_opportunities": "总机会数",
    "used": "已使用",
    "missed": "错失",
    "good_actions": "良好防守",
    "poor_actions": "防守不佳",
    "majority_behavior": "主要行为",
    "Attack Majority": "攻击为主",
    "Retreat Majority": "退却为主",
    "Total Count": "总数",
    "Victory Rate %": "胜率 %",
    "Count / Percentage": "数量 / 百分比",
    "Average Attack Distance (m)": "平均攻击距离 (米)",
    "Optimal Distance (2m)": "最佳距离 (2米)",
    "Overall Average": "总体平均: {:.2f}米",
    "Retreat Distance (m)": "平均退却距离 (米)",
    "Percentage": "百分比",
    "Number of Bouts": "回合数",
    "Good": "良好",
    "Poor": "不佳",
    "Winner": "获胜者",
    "Loser": "失败者",
    "Good Defense": "良好防守 ({})",
    "Poor Defense": "防守不佳 ({})",
    "Used": "已使用 ({})",
    "Missed": "错失 ({})",
    "Safe Distance\nMaintaining": "保持安全\n距离",
    "Consistent Spacing": "间距一致"
}


# Set up matplotlib style
plt.style.use('default')
sns.set_palette("husl")

def create_output_dirs(base_dir):
    """Create output directories for left and right fencer plots."""
    left_dir = os.path.join(base_dir, 'Fencer_Left')
    right_dir = os.path.join(base_dir, 'Fencer_Right')
    
    os.makedirs(left_dir, exist_ok=True)
    os.makedirs(right_dir, exist_ok=True)
    
    return left_dir, right_dir

def plot_attack_type_victory(data, ax, title):
    """Graph 1: Attack type analysis with victory correlation"""
    if not data:
        ax.text(0.5, 0.5, '无攻击类型数据', ha='center', va='center', transform=ax.transAxes)
        ax.set_title(title)
        return
    
    # Count attack types and victories
    attack_counts = {}
    attack_victories = {}
    
    for entry in data:
        attack_type = entry['attack_type']
        is_winner = entry['is_winner']
        
        if attack_type not in attack_counts:
            attack_counts[attack_type] = 0
            attack_victories[attack_type] = 0
        
        attack_counts[attack_type] += 1
        if is_winner:
            attack_victories[attack_type] += 1
    
    # Calculate victory percentages
    attack_types = list(attack_counts.keys())
    counts = [attack_counts[at] for at in attack_types]
    victory_percentages = [(attack_victories[at] / attack_counts[at] * 100) if attack_counts[at] > 0 else 0 
                          for at in attack_types]
    
    # Create grouped bar chart
    x = np.arange(len(attack_types))
    width = 0.35
    
    bars1 = ax.bar(x - width/2, counts, width, label=LABEL_TRANSLATIONS['Total Count'], alpha=0.8, color='#2196F3')
    bars2 = ax.bar(x + width/2, victory_percentages, width, label=LABEL_TRANSLATIONS['Victory Rate %'], alpha=0.8, color='#4CAF50')
    
    # Add value labels on bars
    for bar in bars1:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.1,
               f'{int(height)}', ha='center', va='bottom', fontweight='bold')
    
    for bar in bars2:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 1,
               f'{height:.1f}%', ha='center', va='bottom', fontweight='bold')
    
    ax.set_xlabel(LABEL_TRANSLATIONS['attack_type'])
    ax.set_ylabel(LABEL_TRANSLATIONS['Count / Percentage'])
    ax.set_title(title)
    ax.set_xticks(x)
    ax.set_xticklabels(attack_types, rotation=45, ha='right')
    ax.legend()
    ax.grid(True, alpha=0.3)

def plot_tempo_type_victory(data, ax, title):
    """Graph 2: Tempo type analysis with victory correlation"""
    if not data:
        ax.text(0.5, 0.5, '无节奏类型数据', ha='center', va='center', transform=ax.transAxes)
        ax.set_title(title)
        return
    
    # Count tempo types and victories
    tempo_counts = {}
    tempo_victories = {}
    
    for entry in data:
        tempo_type = entry['tempo_type']
        is_winner = entry['is_winner']
        
        if tempo_type not in tempo_counts:
            tempo_counts[tempo_type] = 0
            tempo_victories[tempo_type] = 0
        
        tempo_counts[tempo_type] += 1
        if is_winner:
            tempo_victories[tempo_type] += 1
    
    # Calculate victory percentages
    tempo_types = list(tempo_counts.keys())
    counts = [tempo_counts[tt] for tt in tempo_types]
    victory_percentages = [(tempo_victories[tt] / tempo_counts[tt] * 100) if tempo_counts[tt] > 0 else 0 
                          for tt in tempo_types]
    
    # Create grouped bar chart
    x = np.arange(len(tempo_types))
    width = 0.35
    
    bars1 = ax.bar(x - width/2, counts, width, label=LABEL_TRANSLATIONS['Total Count'], alpha=0.8, color='#FF9800')
    bars2 = ax.bar(x + width/2, victory_percentages, width, label=LABEL_TRANSLATIONS['Victory Rate %'], alpha=0.8, color='#4CAF50')
    
    # Add value labels on bars
    for bar in bars1:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.1,
               f'{int(height)}', ha='center', va='bottom', fontweight='bold')
    
    for bar in bars2:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 1,
               f'{height:.1f}%', ha='center', va='bottom', fontweight='bold')
    
    ax.set_xlabel(LABEL_TRANSLATIONS['tempo_type'])
    ax.set_ylabel(LABEL_TRANSLATIONS['Count / Percentage'])
    ax.set_title(title)
    ax.set_xticks(x)
    ax.set_xticklabels(tempo_types, rotation=45, ha='right')
    ax.legend()
    ax.grid(True, alpha=0.3)

def plot_attack_distance_analysis(data, ax, title):
    """Graph 3: Average attack distance per bout with 2m reference line"""
    if not data:
        ax.text(0.5, 0.5, '无攻击距离数据', ha='center', va='center', transform=ax.transAxes)
        ax.set_title(title)
        return
    
    # Extract data
    bout_indices = [entry['bout_idx'] for entry in data]
    avg_distances = [entry['avg_distance'] for entry in data]
    
    # Calculate overall statistics
    overall_avg_distance = sum(avg_distances) / len(avg_distances)
    
    # Create bar chart for each bout
    colors = ['#4CAF50' if entry['is_winner'] else '#FF5722' for entry in data]
    bars = ax.bar(range(len(bout_indices)), avg_distances, color=colors, alpha=0.7)
    
    # Add distance labels on bars
    for i, (bar, dist) in enumerate(zip(bars, avg_distances)):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.05,
               f'{dist:.2f}m', ha='center', va='bottom', fontweight='bold')
    
    # Add 2-meter reference line
    ax.axhline(y=2.0, color='red', linestyle='--', linewidth=2, alpha=0.8,
              label=LABEL_TRANSLATIONS['Optimal Distance (2m)'])
    
    # Add overall average line
    ax.axhline(y=overall_avg_distance, color='black', linestyle=':', alpha=0.8,
              label=LABEL_TRANSLATIONS['Overall Average'].format(overall_avg_distance))
    
    ax.set_xlabel(LABEL_TRANSLATIONS['bout_idx'])
    ax.set_ylabel(LABEL_TRANSLATIONS['Average Attack Distance (m)'])
    ax.set_title(title)
    ax.set_xticks(range(len(bout_indices)))
    ax.set_xticklabels([f"Bout {i+1}" for i in bout_indices], rotation=45, ha='right')
    ax.legend()
    ax.grid(True, alpha=0.3)

def plot_counter_opportunities(data, ax, title):
    """Graph 4: Counter opportunities used, missed, and averages"""
    if not data:
        ax.text(0.5, 0.5, '无反击机会数据', ha='center', va='center', transform=ax.transAxes)
        ax.set_title(title)
        return
    
    # Calculate totals
    total_opportunities = sum([entry['total_opportunities'] for entry in data])
    total_used = sum([entry['used'] for entry in data])
    total_missed = sum([entry['missed'] for entry in data])
    
    if total_opportunities == 0:
        ax.text(0.5, 0.5, '未发现反击机会', ha='center', va='center', transform=ax.transAxes)
        ax.set_title(title)
        return
    
    # Calculate percentages
    used_percentage = (total_used / total_opportunities) * 100
    missed_percentage = (total_missed / total_opportunities) * 100
    
    # Create pie chart
    sizes = [used_percentage, missed_percentage]
    labels = [LABEL_TRANSLATIONS['Used'].format(total_used), LABEL_TRANSLATIONS['Missed'].format(total_missed)]
    colors = ['#4CAF50', '#FF5722']
    explode = (0.05, 0.05)  # slightly separate slices
    
    wedges, texts, autotexts = ax.pie(sizes, explode=explode, labels=labels, colors=colors,
                                     autopct='%1.1f%%', shadow=True, startangle=90)
    
    # Enhance text
    for autotext in autotexts:
        autotext.set_color('white')
        autotext.set_fontweight('bold')
    
    ax.set_title(title)
    
    # Add summary text
    summary_text = f"{LABEL_TRANSLATIONS['total_opportunities']}: {total_opportunities}\n{LABEL_TRANSLATIONS['Victory Rate %']}: {used_percentage:.1f}%"
    ax.text(0.02, 0.98, summary_text, transform=ax.transAxes, fontsize=10,
           verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

def plot_retreat_quality_analysis(data, ax, title):
    """Graph 5: Safe distance and spacing consistency in retreats"""
    if not data:
        ax.text(0.5, 0.5, '无退却质量数据', ha='center', va='center', transform=ax.transAxes)
        ax.set_title(title)
        return
    
    # Calculate overall statistics
    total_retreats = sum([entry['total_retreats'] for entry in data])
    total_safe_distance = sum([entry['safe_distance'] for entry in data])
    total_consistent_spacing = sum([entry['consistent_spacing'] for entry in data])
    
    safe_distance_percentage = (total_safe_distance / total_retreats * 100) if total_retreats > 0 else 0
    consistent_spacing_percentage = (total_consistent_spacing / total_retreats * 100) if total_retreats > 0 else 0
    
    # Create stacked bar chart
    categories = [LABEL_TRANSLATIONS['Safe Distance\nMaintaining'], LABEL_TRANSLATIONS['Consistent Spacing']]
    good_percentages = [safe_distance_percentage, consistent_spacing_percentage]
    poor_percentages = [100 - safe_distance_percentage, 100 - consistent_spacing_percentage]
    
    x = np.arange(len(categories))
    width = 0.6
    
    bars1 = ax.bar(x, good_percentages, width, label=LABEL_TRANSLATIONS['Good'], color='#4CAF50', alpha=0.8)
    bars2 = ax.bar(x, poor_percentages, width, bottom=good_percentages, label=LABEL_TRANSLATIONS['Poor'], color='#FF5722', alpha=0.8)
    
    # Add percentage labels
    for i, (good, poor) in enumerate(zip(good_percentages, poor_percentages)):
        ax.text(i, good/2, f'{good:.1f}%', ha='center', va='center', color='white', fontweight='bold')
        ax.text(i, good + poor/2, f'{poor:.1f}%', ha='center', va='center', color='white', fontweight='bold')
    
    ax.set_ylabel(LABEL_TRANSLATIONS['Percentage'])
    ax.set_title(title)
    ax.set_xticks(x)
    ax.set_xticklabels(categories)
    ax.legend()
    ax.set_ylim(0, 100)
    ax.grid(True, alpha=0.3)

def plot_retreat_distance_analysis(data, ax, title):
    """Graph 6: Average retreat distance per bout with 2m reference line"""
    if not data:
        ax.text(0.5, 0.5, '无退却距离数据', ha='center', va='center', transform=ax.transAxes)
        ax.set_title(title)
        return
    
    # Extract data
    bout_indices = [entry['bout_idx'] for entry in data]
    avg_distances = [entry['avg_distance'] for entry in data]
    distance_variances = [entry['distance_variance'] for entry in data]
    
    # Calculate overall statistics
    overall_avg_distance = sum(avg_distances) / len(avg_distances)
    overall_avg_variance = sum(distance_variances) / len(distance_variances)
    
    # Create bar chart for each bout
    colors = ['#4CAF50' if entry['is_winner'] else '#FF5722' for entry in data]
    bars = ax.bar(range(len(bout_indices)), avg_distances, color=colors, alpha=0.7)
    
    # Add distance labels on bars
    for i, (bar, dist, var) in enumerate(zip(bars, avg_distances, distance_variances)):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.05,
               f'{dist:.2f}m\n(σ²={var:.3f})', ha='center', va='bottom', fontweight='bold', fontsize=8)
    
    # Add 2-meter reference line
    ax.axhline(y=2.0, color='red', linestyle='--', linewidth=2, alpha=0.8,
              label=LABEL_TRANSLATIONS['Optimal Distance (2m)'])
    
    # Add overall average line
    ax.axhline(y=overall_avg_distance, color='black', linestyle=':', alpha=0.8,
              label=LABEL_TRANSLATIONS['Overall Average'].format(overall_avg_distance))
    
    ax.set_xlabel(LABEL_TRANSLATIONS['bout_idx'])
    ax.set_ylabel(LABEL_TRANSLATIONS['Retreat Distance (m)'])
    ax.set_title(title)
    ax.set_xticks(range(len(bout_indices)))
    ax.set_xticklabels([f"Bout {i+1}" for i in bout_indices], rotation=45, ha='right')
    ax.legend()
    ax.grid(True, alpha=0.3)

def plot_defensive_quality(data, ax, title):
    """Graph 7: Defensive quality good vs not good"""
    if not data:
        ax.text(0.5, 0.5, '无防守质量数据', ha='center', va='center', transform=ax.transAxes)
        ax.set_title(title)
        return
    
    # Calculate totals
    total_good = sum([entry['good_actions'] for entry in data])
    total_poor = sum([entry['poor_actions'] for entry in data])
    total_actions = total_good + total_poor
    
    if total_actions == 0:
        ax.text(0.5, 0.5, 'No defensive actions found', ha='center', va='center', transform=ax.transAxes)
        ax.set_title(title)
        return
    
    # Calculate percentages
    good_percentage = (total_good / total_actions) * 100
    poor_percentage = (total_poor / total_actions) * 100
    
    # Create pie chart
    sizes = [good_percentage, poor_percentage]
    labels = [LABEL_TRANSLATIONS['Good Defense'].format(total_good), LABEL_TRANSLATIONS['Poor Defense'].format(total_poor)]
    colors = ['#4CAF50', '#FF5722']
    explode = (0.05, 0.05)  # slightly separate slices
    
    wedges, texts, autotexts = ax.pie(sizes, explode=explode, labels=labels, colors=colors,
                                     autopct='%1.1f%%', shadow=True, startangle=90)
    
    # Enhance text
    for autotext in autotexts:
        autotext.set_color('white')
        autotext.set_fontweight('bold')
    
    ax.set_title(title)
    
    # Add summary text
    summary_text = f"{LABEL_TRANSLATIONS['total_opportunities']}: {total_actions}\n{LABEL_TRANSLATIONS['Good Defense'][:-3]} Rate: {good_percentage:.1f}%" # Remove placeholder from label
    ax.text(0.02, 0.98, summary_text, transform=ax.transAxes, fontsize=10,
           verticalalignment='top', bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.5))

def plot_bout_outcome_analysis(data, ax, title):
    """Graph 8: Bout outcome analysis - attack vs retreat victory patterns"""
    if not data:
        ax.text(0.5, 0.5, '无回合结果数据', ha='center', va='center', transform=ax.transAxes)
        ax.set_title(title)
        return
    
    # Separate winners and losers
    winners = [entry for entry in data if entry['is_winner']]
    losers = [entry for entry in data if not entry['is_winner']]
    
    # Count behavior patterns
    winner_attack = len([w for w in winners if w['majority_behavior'] == 'attack'])
    winner_retreat = len([w for w in winners if w['majority_behavior'] == 'retreat'])
    loser_attack = len([l for l in losers if l['majority_behavior'] == 'attack'])
    loser_retreat = len([l for l in losers if l['majority_behavior'] == 'retreat'])
    
    # Create grouped bar chart
    categories = [LABEL_TRANSLATIONS['Attack Majority'], LABEL_TRANSLATIONS['Retreat Majority']]
    winner_counts = [winner_attack, winner_retreat]
    loser_counts = [loser_attack, loser_retreat]
    
    x = np.arange(len(categories))
    width = 0.35
    
    bars1 = ax.bar(x - width/2, winner_counts, width, label=LABEL_TRANSLATIONS['Winner'], color='#4CAF50', alpha=0.8)
    bars2 = ax.bar(x + width/2, loser_counts, width, label=LABEL_TRANSLATIONS['Loser'], color='#FF5722', alpha=0.8)
    
    # Add value labels on bars
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            if height > 0:
                ax.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                       f'{int(height)}', ha='center', va='bottom', fontweight='bold')
    
    ax.set_xlabel(LABEL_TRANSLATIONS['majority_behavior'])
    ax.set_ylabel(LABEL_TRANSLATIONS['Number of Bouts'])
    ax.set_title(title)
    ax.set_xticks(x)
    ax.set_xticklabels(categories)
    ax.legend()
    ax.grid(True, alpha=0.3)

def save_fencer_analysis_plots(fencer_data, fencer_side, output_dir):
    """Save all 8 analysis plots for a specific fencer as individual images"""
    
    plot_files = {}
    fencer_title = TITLE_TRANSLATIONS[fencer_side.title() + ' Fencer']
    
    # Plot 1: Attack Type Victory Analysis
    fig1, ax1 = plt.subplots(figsize=(10, 6))
    plot_attack_type_victory(fencer_data['attack_types'], ax1, 
                           f"{fencer_title}: {TITLE_TRANSLATIONS['Attack Type & Victory Analysis']}")
    plt.tight_layout()
    attack_type_path = os.path.join(output_dir, f'{fencer_side}_attack_type_analysis.png')
    plt.savefig(attack_type_path, dpi=300, bbox_inches='tight')
    plot_files['attack_type_analysis'] = attack_type_path
    plt.close()
    
    # Plot 2: Tempo Type Victory Analysis
    fig2, ax2 = plt.subplots(figsize=(10, 6))
    plot_tempo_type_victory(fencer_data['tempo_types'], ax2,
                          f"{fencer_title}: {TITLE_TRANSLATIONS['Tempo Type & Victory Analysis']}")
    plt.tight_layout()
    tempo_type_path = os.path.join(output_dir, f'{fencer_side}_tempo_type_analysis.png')
    plt.savefig(tempo_type_path, dpi=300, bbox_inches='tight')
    plot_files['tempo_type_analysis'] = tempo_type_path
    plt.close()
    
    # Plot 3: Attack Distance Analysis
    fig3, ax3 = plt.subplots(figsize=(10, 6))
    plot_attack_distance_analysis(fencer_data['attack_distances'], ax3,
                             f"{fencer_title}: {TITLE_TRANSLATIONS['Attack Distance Analysis']}")
    plt.tight_layout()
    attack_distance_path = os.path.join(output_dir, f'{fencer_side}_attack_distance_analysis.png')
    plt.savefig(attack_distance_path, dpi=300, bbox_inches='tight')
    plot_files['attack_distance_analysis'] = attack_distance_path
    plt.close()
    
    # Plot 4: Counter Opportunities Analysis
    fig4, ax4 = plt.subplots(figsize=(10, 6))
    plot_counter_opportunities(fencer_data['counter_opportunities'], ax4,
                             f"{fencer_title}: {TITLE_TRANSLATIONS['Counter Opportunity Analysis']}")
    plt.tight_layout()
    counter_opp_path = os.path.join(output_dir, f'{fencer_side}_counter_opportunities.png')
    plt.savefig(counter_opp_path, dpi=300, bbox_inches='tight')
    plot_files['counter_opportunities'] = counter_opp_path
    plt.close()
    
    # Plot 5: Retreat Quality Analysis
    fig5, ax5 = plt.subplots(figsize=(10, 6))
    plot_retreat_quality_analysis(fencer_data['retreat_quality'], ax5,
                                 f"{fencer_title}: {TITLE_TRANSLATIONS['Retreat Quality Analysis']}")
    plt.tight_layout()
    retreat_quality_path = os.path.join(output_dir, f'{fencer_side}_retreat_quality.png')
    plt.savefig(retreat_quality_path, dpi=300, bbox_inches='tight')
    plot_files['retreat_quality'] = retreat_quality_path
    plt.close()
    
    # Plot 6: Retreat Distance Analysis
    fig6, ax6 = plt.subplots(figsize=(10, 6))
    plot_retreat_distance_analysis(fencer_data['retreat_distances'], ax6,
                                 f"{fencer_title}: {TITLE_TRANSLATIONS['Retreat Distance Analysis']}")
    plt.tight_layout()
    retreat_distance_path = os.path.join(output_dir, f'{fencer_side}_retreat_distance.png')
    plt.savefig(retreat_distance_path, dpi=300, bbox_inches='tight')
    plot_files['retreat_distance'] = retreat_distance_path
    plt.close()
    
    # Plot 7: Defensive Quality Analysis
    fig7, ax7 = plt.subplots(figsize=(10, 6))
    plot_defensive_quality(fencer_data['defensive_quality'], ax7,
                          f"{fencer_title}: {TITLE_TRANSLATIONS['Defensive Quality Analysis']}")
    plt.tight_layout()
    defensive_quality_path = os.path.join(output_dir, f'{fencer_side}_defensive_quality.png')
    plt.savefig(defensive_quality_path, dpi=300, bbox_inches='tight')
    plot_files['defensive_quality'] = defensive_quality_path
    plt.close()
    
    # Plot 8: Bout Outcome Analysis
    fig8, ax8 = plt.subplots(figsize=(10, 6))
    plot_bout_outcome_analysis(fencer_data['bout_outcomes'], ax8,
                              f"{fencer_title}: {TITLE_TRANSLATIONS['Bout Outcome Analysis']}")
    plt.tight_layout()
    bout_outcome_path = os.path.join(output_dir, f'{fencer_side}_bout_outcome.png')
    plt.savefig(bout_outcome_path, dpi=300, bbox_inches='tight')
    plot_files['bout_outcome'] = bout_outcome_path
    plt.close()
    
    logging.info(f"Saved 8 individual analysis plots for {fencer_side} fencer in {output_dir}")
    logging.info(f"Generated plots: {list(plot_files.keys())}")
    return plot_files

def save_all_plots(left_data, right_data, base_output_dir):
    """
    Save all analysis plots for both fencers in separate directories.
    Generate graph-specific analysis for each plot.
    Returns dictionary with plot file paths and analysis organized by fencer.
    """
    from .graph_analysis import generate_graph_analysis
    
    # Create output directories
    left_dir, right_dir = create_output_dirs(base_output_dir)
    
    # Save plots for each fencer
    all_plot_files = {}
    all_analysis = {}
    
    # Left fencer plots
    left_plots = save_fencer_analysis_plots(left_data, 'left', left_dir)
    all_plot_files['Fencer_Left'] = left_plots
    
    # Right fencer plots  
    right_plots = save_fencer_analysis_plots(right_data, 'right', right_dir)
    all_plot_files['Fencer_Right'] = right_plots
    
    # Generate graph-specific analysis for each of the 8 graph types
    graph_analysis = {}
    
    try:
        graph_types = [
            'attack_type_analysis',
            'tempo_type_analysis', 
            'attack_distance_analysis',
            'counter_opportunities',
            'retreat_quality',
            'retreat_distance',
            'defensive_quality',
            'bout_outcome'
        ]
        
        for graph_type in graph_types:
            # Map graph type to data key
            data_key_map = {
                'attack_type_analysis': 'attack_types',
                'tempo_type_analysis': 'tempo_types',
                'attack_distance_analysis': 'attack_distances',
                'counter_opportunities': 'counter_opportunities',
                'retreat_quality': 'retreat_quality',
                'retreat_distance': 'retreat_distances',
                'defensive_quality': 'defensive_quality',
                'bout_outcome': 'bout_outcomes'
            }
            
            data_key = data_key_map[graph_type]
            left_graph_data = left_data.get(data_key, [])
            right_graph_data = right_data.get(data_key, [])
            
            # Generate analysis for this specific graph
            analysis = generate_graph_analysis(graph_type, left_graph_data, right_graph_data)
            graph_analysis[graph_type] = analysis
            
            logging.info(f"Generated analysis for {graph_type}")
        
        # Save all graph analysis to JSON file
        analysis_file = os.path.join(base_output_dir, 'graph_analysis.json')
        with open(analysis_file, 'w', encoding='utf-8') as f:
            json.dump(graph_analysis, f, ensure_ascii=False, indent=2)
        
        all_analysis['graph_analysis'] = graph_analysis
        all_analysis['analysis_file'] = analysis_file
        
        logging.info(f"Saved graph analysis to {analysis_file}")
        
    except Exception as e:
        logging.error(f"Error generating graph analysis: {e}")
        all_analysis['graph_analysis'] = {}
    
    logging.info(f"Generated all 8 analysis plots for both fencers")
    logging.info(f"Left fencer plots: {len(left_plots)} files in {left_dir}")
    logging.info(f"Right fencer plots: {len(right_plots)} files in {right_dir}")
    
    return {
        'plot_files': all_plot_files,
        'analysis': all_analysis
    } 