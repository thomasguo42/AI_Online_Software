import matplotlib.pyplot as plt
import numpy as np
import os
import logging
from typing import Dict, Any, List

# Set up matplotlib for Chinese fonts
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

def create_inbox_analysis_charts(inbox_data: Dict[str, Any], output_dir: str) -> Dict[str, str]:
    """
    Create visualization charts specifically for In-Box analysis.
    
    Args:
        inbox_data: In-Box analysis data
        output_dir: Directory to save charts
    
    Returns:
        Dictionary with paths to created chart files
    """
    os.makedirs(output_dir, exist_ok=True)
    chart_paths = {}
    
    if not inbox_data or not inbox_data.get('bouts'):
        logging.warning("No In-Box data available for visualization")
        return chart_paths
    
    try:
        bouts = inbox_data['bouts']
        
        # 1. Attack Type Distribution Chart
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
        
        # Left fencer attack types (use fields produced by inbox processing)
        left_attack_types = [
            bout.get('left_fencer', {}).get('attack_type_last_specific')
            or bout.get('left_fencer', {}).get('attack_type_last')
            or '无攻击'
            for bout in bouts
        ]
        left_type_counts = {}
        for atype in left_attack_types:
            left_type_counts[atype] = left_type_counts.get(atype, 0) + 1
        
        # Right fencer attack types
        right_attack_types = [
            bout.get('right_fencer', {}).get('attack_type_last_specific')
            or bout.get('right_fencer', {}).get('attack_type_last')
            or '无攻击'
            for bout in bouts
        ]
        right_type_counts = {}
        for atype in right_attack_types:
            right_type_counts[atype] = right_type_counts.get(atype, 0) + 1
        
        # Plot attack type distributions
        colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7', '#DDA0DD']
        
        if left_type_counts:
            ax1.pie(left_type_counts.values(), labels=left_type_counts.keys(), autopct='%1.1f%%', colors=colors)
            ax1.set_title('左侧击剑手攻击类型分布', fontsize=12, fontweight='bold')
        
        if right_type_counts:
            ax2.pie(right_type_counts.values(), labels=right_type_counts.keys(), autopct='%1.1f%%', colors=colors)
            ax2.set_title('右侧击剑手攻击类型分布', fontsize=12, fontweight='bold')
        
        plt.suptitle('In-Box 攻击类型分析', fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        attack_type_path = os.path.join(output_dir, 'inbox_attack_types.png')
        plt.savefig(attack_type_path, dpi=300, bbox_inches='tight')
        plt.close()
        chart_paths['attack_types'] = attack_type_path
        
        # 2. Velocity vs Acceleration Scatter Plot
        fig, ax = plt.subplots(figsize=(10, 8))
        
        left_velocities = [bout.get('left_fencer', {}).get('velocity', {}).get('mean', 0) for bout in bouts]
        left_accelerations = [bout.get('left_fencer', {}).get('acceleration', {}).get('mean', 0) for bout in bouts]
        right_velocities = [bout.get('right_fencer', {}).get('velocity', {}).get('mean', 0) for bout in bouts]
        right_accelerations = [bout.get('right_fencer', {}).get('acceleration', {}).get('mean', 0) for bout in bouts]
        
        ax.scatter(left_velocities, left_accelerations, c='#FF6B6B', alpha=0.7, s=60, label='左侧击剑手', edgecolors='white', linewidth=1)
        ax.scatter(right_velocities, right_accelerations, c='#4ECDC4', alpha=0.7, s=60, label='右侧击剑手', edgecolors='white', linewidth=1)
        
        ax.set_xlabel('平均速度 (m/s)', fontsize=12)
        ax.set_ylabel('平均加速度 (m/s²)', fontsize=12)
        ax.set_title('In-Box 速度与加速度关系', fontsize=14, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        velocity_accel_path = os.path.join(output_dir, 'inbox_velocity_acceleration.png')
        plt.savefig(velocity_accel_path, dpi=300, bbox_inches='tight')
        plt.close()
        chart_paths['velocity_acceleration'] = velocity_accel_path
        
        # 3. Initial Step Timing Analysis
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
        
        left_step_times = [bout.get('left_fencer', {}).get('initial_step', {}).get('onset_time_s', 0) for bout in bouts]
        right_step_times = [bout.get('right_fencer', {}).get('initial_step', {}).get('onset_time_s', 0) for bout in bouts]
        
        ax1.hist(left_step_times, bins=10, alpha=0.7, color='#FF6B6B', edgecolor='white')
        ax1.set_xlabel('初始步时间 (秒)', fontsize=10)
        ax1.set_ylabel('频次', fontsize=10)
        ax1.set_title('左侧击剑手初始步时间分布', fontsize=12, fontweight='bold')
        ax1.grid(True, alpha=0.3)
        
        ax2.hist(right_step_times, bins=10, alpha=0.7, color='#4ECDC4', edgecolor='white')
        ax2.set_xlabel('初始步时间 (秒)', fontsize=10)
        ax2.set_ylabel('频次', fontsize=10)
        ax2.set_title('右侧击剑手初始步时间分布', fontsize=12, fontweight='bold')
        ax2.grid(True, alpha=0.3)
        
        plt.suptitle('In-Box 初始步时间分析', fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        step_timing_path = os.path.join(output_dir, 'inbox_step_timing.png')
        plt.savefig(step_timing_path, dpi=300, bbox_inches='tight')
        plt.close()
        chart_paths['step_timing'] = step_timing_path
        
        logging.info(f"Created {len(chart_paths)} In-Box analysis charts")
        
    except Exception as e:
        logging.error(f"Error creating In-Box charts: {e}")
    
    return chart_paths

def create_touch_category_charts(touch_stats: Dict[str, Any], output_dir: str) -> Dict[str, str]:
    """
    Create visualization charts for bout category statistics.
    
    Args:
        touch_stats: Touch statistics dictionary from classification
        output_dir: Directory to save the charts
    
    Returns:
        Dictionary with paths to created chart files
    """
    os.makedirs(output_dir, exist_ok=True)
    chart_paths = {}
    
    try:
        # 1. Create overall category distribution chart
        chart_paths['category_distribution'] = create_category_distribution_chart(touch_stats, output_dir)
        
        # 2. Create win rate by category chart
        chart_paths['win_rates'] = create_win_rate_chart(touch_stats, output_dir)
        
        # 3. Create detailed statistics table chart
        chart_paths['detailed_stats'] = create_detailed_stats_chart(touch_stats, output_dir)
        
        logging.info(f"Created {len(chart_paths)} bout category charts")
        
    except Exception as e:
        logging.error(f"Error creating bout category charts: {e}")
    
    return chart_paths

def create_category_distribution_chart(touch_stats: Dict[str, Any], output_dir: str) -> str:
    """Create a bar chart showing the distribution of touch categories."""
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
    
    categories = ['in_box', 'attack', 'defense']
    category_labels = ['对攻', '进攻', '防守']
    colors = ['#FFA500', '#FF6B6B', '#4ECDC4']
    
    # Left fencer data
    left_counts = [touch_stats['left_fencer'][cat]['count'] for cat in categories]
    ax1.bar(category_labels, left_counts, color=colors, alpha=0.8)
    ax1.set_title('左侧击剑手 - 回合类型分布', fontsize=14, fontweight='bold')
    ax1.set_ylabel('回合数', fontsize=12)
    ax1.set_xlabel('回合类型', fontsize=12)
    
    # Add value labels on bars
    for i, v in enumerate(left_counts):
        ax1.text(i, v + 0.1, str(v), ha='center', va='bottom', fontweight='bold')
    
    # Right fencer data
    right_counts = [touch_stats['right_fencer'][cat]['count'] for cat in categories]
    ax2.bar(category_labels, right_counts, color=colors, alpha=0.8)
    ax2.set_title('右侧击剑手 - 回合类型分布', fontsize=14, fontweight='bold')
    ax2.set_ylabel('回合数', fontsize=12)
    ax2.set_xlabel('回合类型', fontsize=12)
    
    # Add value labels on bars
    for i, v in enumerate(right_counts):
        ax2.text(i, v + 0.1, str(v), ha='center', va='bottom', fontweight='bold')
    
    plt.tight_layout()
    chart_path = os.path.join(output_dir, 'bout_category_distribution.png')
    plt.savefig(chart_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    
    return chart_path

def create_win_rate_chart(touch_stats: Dict[str, Any], output_dir: str) -> str:
    """Create a chart showing win rates by touch category."""
    
    fig, ax = plt.subplots(figsize=(14, 8))
    
    categories = ['in_box', 'attack', 'defense']
    category_labels = ['对攻', '进攻', '防守']
    
    # Extract win rates
    left_win_rates = [touch_stats['left_fencer'][cat]['win_rate'] for cat in categories]
    right_win_rates = [touch_stats['right_fencer'][cat]['win_rate'] for cat in categories]
    
    x = np.arange(len(category_labels))
    width = 0.35
    
    bars1 = ax.bar(x - width/2, left_win_rates, width, label='左侧击剑手', 
                   color='#1f77b4', alpha=0.8)
    bars2 = ax.bar(x + width/2, right_win_rates, width, label='右侧击剑手', 
                   color='#ff7f0e', alpha=0.8)
    
    ax.set_title('各回合类型胜率对比', fontsize=16, fontweight='bold', pad=20)
    ax.set_ylabel('胜率 (%)', fontsize=12)
    ax.set_xlabel('回合类型', fontsize=12)
    ax.set_xticks(x)
    ax.set_xticklabels(category_labels)
    ax.legend(fontsize=11)
    ax.set_ylim(0, 100)
    
    # Add value labels on bars
    def add_value_labels(bars, values):
        for bar, value in zip(bars, values):
            height = bar.get_height()
            ax.annotate(f'{value:.1f}%',
                       xy=(bar.get_x() + bar.get_width() / 2, height),
                       xytext=(0, 3),
                       textcoords="offset points",
                       ha='center', va='bottom',
                       fontsize=10, fontweight='bold')
    
    add_value_labels(bars1, left_win_rates)
    add_value_labels(bars2, right_win_rates)
    
    ax.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    
    chart_path = os.path.join(output_dir, 'bout_category_win_rates.png')
    plt.savefig(chart_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    
    return chart_path

def create_detailed_stats_chart(touch_stats: Dict[str, Any], output_dir: str) -> str:
    """Create a detailed statistics table as a chart."""
    
    fig, ax = plt.subplots(figsize=(16, 10))
    ax.axis('tight')
    ax.axis('off')
    
    # Prepare data for table
    categories = ['in_box', 'attack', 'defense']
    category_labels = ['对攻', '进攻', '防守']
    
    table_data = []
    
    # Header row
    table_data.append(['击剑手', '触剑类型', '总回合数', '胜利数', '失败数', '胜率 (%)'])
    
    # Left fencer data
    for i, cat in enumerate(categories):
        stats = touch_stats['left_fencer'][cat]
        table_data.append([
            '左侧击剑手' if i == 0 else '',
            category_labels[i],
            str(stats['count']),
            str(stats['wins']),
            str(stats['losses']),
            f"{stats['win_rate']:.1f}%"
        ])
    
    # Add separator row
    table_data.append(['', '', '', '', '', ''])
    
    # Right fencer data
    for i, cat in enumerate(categories):
        stats = touch_stats['right_fencer'][cat]
        table_data.append([
            '右侧击剑手' if i == 0 else '',
            category_labels[i],
            str(stats['count']),
            str(stats['wins']),
            str(stats['losses']),
            f"{stats['win_rate']:.1f}%"
        ])
    
    # Create table
    table = ax.table(cellText=table_data, loc='center', cellLoc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(12)
    table.scale(1.2, 2)
    
    # Style the table
    # Header row
    for i in range(len(table_data[0])):
        table[(0, i)].set_facecolor('#4472C4')
        table[(0, i)].set_text_props(weight='bold', color='white')
    
    # Left fencer rows
    for row in range(1, 4):
        for col in range(len(table_data[0])):
            table[(row, col)].set_facecolor('#E7F3FF')
    
    # Right fencer rows
    for row in range(5, 8):
        for col in range(len(table_data[0])):
            table[(row, col)].set_facecolor('#FFF2E7')
    
    # Separator row
    for col in range(len(table_data[0])):
        table[(4, col)].set_facecolor('#F0F0F0')
    
    plt.title('回合类型详细统计表', fontsize=16, fontweight='bold', pad=20)
    plt.tight_layout()
    
    chart_path = os.path.join(output_dir, 'bout_category_detailed_stats.png')
    plt.savefig(chart_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    
    return chart_path

def generate_touch_category_summary(touch_stats: Dict[str, Any]) -> str:
    """
    Generate a text summary of touch category statistics.
    
    Args:
        touch_stats: Touch statistics dictionary
    
    Returns:
        Formatted text summary
    """
    try:
        summary_lines = []
        summary_lines.append("## 回合类型分析总结\n")
        
        for fencer_key, fencer_label in [('left_fencer', '左侧击剑手'), ('right_fencer', '右侧击剑手')]:
            fencer_data = touch_stats[fencer_key]
            summary_lines.append(f"### {fencer_label}:")
            
            total_bouts = fencer_data['total_bouts']
            summary_lines.append(f"- **总回合数**: {total_bouts}")
            
            # Category breakdown
            for cat, cat_label in [('in_box', '对攻'), ('attack', '进攻'), ('defense', '防守')]:
                cat_data = fencer_data[cat]
                count = cat_data['count']
                wins = cat_data['wins']
                losses = cat_data['losses']
                win_rate = cat_data['win_rate']
                
                if count > 0:
                    percentage = (count / total_bouts * 100) if total_bouts > 0 else 0
                    summary_lines.append(f"- **{cat_label}**: {count}回合 ({percentage:.1f}%) - "
                                        f"胜{wins}负{losses} (胜率: {win_rate:.1f}%)")
            
            summary_lines.append("")
        
        # Overall comparison
        summary_lines.append("### 对比分析:")
        left_data = touch_stats['left_fencer']
        right_data = touch_stats['right_fencer']
        
        # Find strongest categories for each fencer
        categories = ['in_box', 'attack', 'defense']
        cat_labels = {'in_box': '对攻', 'attack': '进攻', 'defense': '防守'}
        
        left_best_cat = max(categories, key=lambda x: left_data[x]['win_rate'] if left_data[x]['count'] > 0 else 0)
        right_best_cat = max(categories, key=lambda x: right_data[x]['win_rate'] if right_data[x]['count'] > 0 else 0)
        
        summary_lines.append(f"- 左侧击剑手最强项: {cat_labels[left_best_cat]} "
                            f"(胜率 {left_data[left_best_cat]['win_rate']:.1f}%)")
        summary_lines.append(f"- 右侧击剑手最强项: {cat_labels[right_best_cat]} "
                            f"(胜率 {right_data[right_best_cat]['win_rate']:.1f}%)")
        
        return "\n".join(summary_lines)
        
    except Exception as e:
        logging.error(f"Error generating touch category summary: {e}")
        return "回合类型分析总结生成失败"


def create_performance_radar_chart(data: List[Dict], category_name: str, output_dir: str) -> str:
    """
    Create a radar chart comparing left and right fencer performance in a specific category.
    """
    os.makedirs(output_dir, exist_ok=True)
    if not data:
        fig, ax = plt.subplots(figsize=(10, 8), subplot_kw=dict(projection='polar'))
        ax.text(0.5, 0.5, f'无{category_name}数据', ha='center', va='center', transform=ax.transAxes)
        chart_path = os.path.join(output_dir, f'{category_name}_performance_radar.png')
        plt.savefig(chart_path, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
        return chart_path

    left_data = []
    right_data = []
    for bout in data:
        # Include only sides actually classified in the category when flags are available
        if 'left_fencer' in bout:
            left_fencer = bout['left_fencer']
            include_left = True
            if category_name == 'attack':
                include_left = bool(bout.get('left_is_attack', False))
            elif category_name == 'defense':
                include_left = bool(bout.get('left_is_defense', False))
            elif category_name == 'in_box':
                include_left = bool(bout.get('left_is_inbox', True))
            if include_left:
                left_data.append(left_fencer)
        
        if 'right_fencer' in bout:
            right_fencer = bout['right_fencer']
            include_right = True
            if category_name == 'attack':
                include_right = bool(bout.get('right_is_attack', False))
            elif category_name == 'defense':
                include_right = bool(bout.get('right_is_defense', False))
            elif category_name == 'in_box':
                include_right = bool(bout.get('right_is_inbox', True))
            if include_right:
                right_data.append(right_fencer)

    # Define metrics (existing code retained above)
    if category_name == 'in_box':
        # Replace 稳定性 with lunge rate and pause rate per spec
        metrics = [
            ('initiative_rate', '弓步率', lambda d: calculate_lunge_rate(d)),
            ('velocity_avg', '平均速度', lambda d: calculate_avg_velocity(d)),
            ('acceleration_avg', '平均加速度', lambda d: calculate_avg_acceleration(d)),
            ('timing_score', '反应时机', lambda d: calculate_timing_score(d)),
            ('arm_extension_rate', '出剑率', lambda d: calculate_arm_extension_rate(d)),
            ('pause_rate', '停顿率', lambda d: calculate_pause_rate(d))
        ]
    elif category_name == 'attack':
        # Replace 战术多样性 and 攻击效率 with lunge rate and arm extension rate
        metrics = [
            ('attack_success', '攻击成功率', lambda d: calculate_attack_success_rate(d)),
            ('velocity_peak', '攻击速度', lambda d: calculate_peak_velocity(d)),
            ('distance_control', '良好距离率', lambda d: calculate_distance_control(d)),
            ('timing_precision', '时机精准度', lambda d: calculate_timing_precision(d)),
            ('lunge_rate', '弓步率', lambda d: calculate_lunge_rate(d)),
            ('arm_extension_rate', '出剑率', lambda d: calculate_arm_extension_rate(d))
        ]
    else:
        # Remove defense velocity/acceleration per spec
        metrics = [
            ('defense_success', '防守成功率', lambda d: calculate_defense_success_rate(d)),
            ('retreat_quality', '后退质量', lambda d: calculate_retreat_quality(d)),
            ('counter_opportunity', '反击机会', lambda d: calculate_counter_opportunity(d)),
            ('distance_safety', '安全距离', lambda d: calculate_distance_safety(d))
        ]

    left_scores = []
    right_scores = []
    metric_labels = []
    for metric_key, metric_label, calc_func in metrics:
        left_score = calc_func(left_data) if left_data else 0
        right_score = calc_func(right_data) if right_data else 0
        left_scores.append(float(left_score))
        right_scores.append(float(right_score))
        metric_labels.append(metric_label)

    # Enhanced comparative scaling for small values
    if category_name == 'in_box':
        # Apply enhanced scaling to acceleration and velocity which often have small values
        for metric_name in ['平均加速度', '平均速度']:
            if metric_name in metric_labels:
                idx = metric_labels.index(metric_name)
                # Get raw values for proper scaling
                def get_raw_values(ds: List[Dict], field_name: str) -> float:
                    vals = []
                    for f in ds:
                        if field_name == 'acceleration':
                            acc_data = f.get('acceleration', {}) or {}
                            val = acc_data.get('mean', acc_data.get('overall_score', 0)) or 0
                        else:  # velocity
                            vel_data = f.get('velocity', {}) or {}
                            val = vel_data.get('mean', vel_data.get('overall_score', 0)) or 0
                        vals.append(float(val))
                    return float(np.mean(vals)) if vals else 0.0
                
                field = 'acceleration' if metric_name == '平均加速度' else 'velocity'
                l_raw = get_raw_values(left_data, field)
                r_raw = get_raw_values(right_data, field)
                
                # Enhanced scaling for small differences
                if l_raw == r_raw == 0:
                    left_scores[idx] = 50
                    right_scores[idx] = 50
                elif abs(l_raw - r_raw) < 0.01:  # Very small difference
                    # Still show relative difference but with meaningful separation
                    if l_raw > r_raw:
                        left_scores[idx] = 65
                        right_scores[idx] = 35
                    elif r_raw > l_raw:
                        left_scores[idx] = 35
                        right_scores[idx] = 65
                    else:
                        left_scores[idx] = 50
                        right_scores[idx] = 50
                else:
                    # Map with expanded range for visibility
                    min_v = min(l_raw, r_raw)
                    max_v = max(l_raw, r_raw)
                    left_scores[idx] = 25 + ((l_raw - min_v) / (max_v - min_v)) * 50
                    right_scores[idx] = 25 + ((r_raw - min_v) / (max_v - min_v)) * 50

    # Plot (existing code retained below)
    fig, ax = plt.subplots(figsize=(12, 10), subplot_kw=dict(projection='polar'))
    angles = np.linspace(0, 2 * np.pi, len(metrics), endpoint=False).tolist()
    angles += angles[:1]
    left_scores += left_scores[:1]
    right_scores += right_scores[:1]
    ax.plot(angles, left_scores, 'o-', linewidth=2, label='左侧击剑手', color='#FF6B6B')
    ax.fill(angles, left_scores, alpha=0.25, color='#FF6B6B')
    ax.plot(angles, right_scores, 'o-', linewidth=2, label='右侧击剑手', color='#4ECDC4')
    ax.fill(angles, right_scores, alpha=0.25, color='#4ECDC4')
    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)
    ax.set_thetagrids(np.degrees(angles[:-1]), metric_labels)
    ax.set_ylim(0, 100)
    ax.set_yticks([20, 40, 60, 80, 100])
    ax.set_yticklabels(['20', '40', '60', '80', '100'])
    ax.grid(True)
    category_titles = {'in_box': '对攻', 'attack': '进攻', 'defense': '防守'}
    plt.title(f"{category_titles.get(category_name, category_name)}回合表现雷达图", size=16, fontweight='bold', pad=20)
    plt.legend(loc='upper right', bbox_to_anchor=(1.2, 1.0))
    plt.tight_layout()
    chart_path = os.path.join(output_dir, f'{category_name}_performance_radar.png')
    plt.savefig(chart_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    return chart_path


def calculate_initiative_rate(fencer_data: List[Dict]) -> float:
    """Calculate initiative rate (percentage of bouts with lunge)."""
    if not fencer_data:
        return 0
    initiative_count = sum(1 for f in fencer_data if f.get('has_launch', False) or f.get('lunge_present', False))
    return (initiative_count / len(fencer_data)) * 100


def calculate_lunge_rate(fencer_data: List[Dict]) -> float:
    """Percentage of bouts with lunge present."""
    if not fencer_data:
        return 0
    count = sum(1 for f in fencer_data if f.get('has_lunge', False) or f.get('lunge', {}).get('present', False))
    return (count / len(fencer_data)) * 100


def calculate_pause_rate(fencer_data: List[Dict]) -> float:
    """Percentage of bouts with pause present."""
    if not fencer_data:
        return 0
    count = 0
    for f in fencer_data:
        pause = f.get('pause', {}) or {}
        # inbox structure uses {'present': bool} under pause
        if isinstance(pause, dict) and pause.get('present'):
            count += 1
        else:
            # fallback: pause_ratio if available
            if float(f.get('pause_ratio', 0) or 0) > 0:
                count += 1
    return (count / len(fencer_data)) * 100


def calculate_avg_velocity(fencer_data: List[Dict]) -> float:
    """Calculate average velocity score (0-100)."""
    if not fencer_data:
        return 0
    velocities = []
    for f in fencer_data:
        vel_data = f.get('velocity', {})
        vel = vel_data.get('mean', vel_data.get('overall_score', 0))
        velocities.append(vel)
    avg_vel = np.mean(velocities) if velocities else 0
    # Normalize to 0-100 scale (assuming max velocity around 5 m/s)
    return min(100, (avg_vel / 5.0) * 100)


def calculate_avg_acceleration(fencer_data: List[Dict]) -> float:
    """Calculate average acceleration score (0-100)."""
    if not fencer_data:
        return 0
    accelerations = []
    for f in fencer_data:
        acc_data = f.get('acceleration', {})
        acc = acc_data.get('mean', acc_data.get('overall_score', 0))
        accelerations.append(acc)
    avg_acc = np.mean(accelerations) if accelerations else 0
    # More reasonable normalization for small values - use 5 m/s² as upper bound for comparison
    return min(100, (avg_acc / 5.0) * 100)


def calculate_timing_score(fencer_data: List[Dict]) -> float:
    """Calculate timing score based on reaction time (0-100)."""
    if not fencer_data:
        return 0
    timing_scores = []
    for f in fencer_data:
        initial_step = f.get('initial_step', {})
        onset_time = initial_step.get('onset_time_s', float('inf'))
        if onset_time != float('inf'):
            # Better timing = lower onset time, convert to score where lower is better
            # Assuming good reaction time is under 0.1s, normalize accordingly
            score = max(0, 100 - (onset_time / 0.1) * 100)
            timing_scores.append(score)
    return np.mean(timing_scores) if timing_scores else 0


def calculate_arm_extension_rate(fencer_data: List[Dict]) -> float:
    """Calculate arm extension rate (percentage of bouts with arm extensions)."""
    if not fencer_data:
        return 0
    extension_count = sum(1 for f in fencer_data if f.get('arm_timing', {}).get('present', False))
    return (extension_count / len(fencer_data)) * 100


def calculate_consistency_score(fencer_data: List[Dict]) -> float:
    """Calculate consistency score based on velocity variance."""
    if not fencer_data:
        return 0
    velocities = []
    for f in fencer_data:
        vel_data = f.get('velocity', {})
        vel = vel_data.get('mean', vel_data.get('overall_score', 0))
        velocities.append(vel)
    if len(velocities) < 2:
        return 50  # Default middle score for single data point
    variance = np.var(velocities)
    # Lower variance = higher consistency, normalize to 0-100
    consistency = max(0, 100 - (variance / 2.0) * 100)
    return min(100, consistency)


def calculate_attack_success_rate(fencer_data: List[Dict]) -> float:
    """Calculate attack success rate."""
    if not fencer_data:
        return 0
    wins = sum(1 for f in fencer_data if f.get('is_winner', False))
    return (wins / len(fencer_data)) * 100


def calculate_peak_velocity(fencer_data: List[Dict]) -> float:
    """Calculate peak/attack velocity score for attack bouts (0-100)."""
    if not fencer_data:
        return 0
    peak_velocities = []
    for f in fencer_data:
        # Prefer attack-specific velocity if present
        if 'attack_velocity' in f and isinstance(f.get('attack_velocity'), (int, float)):
            peak_vel = f.get('attack_velocity', 0) or 0
        else:
            vel_data = f.get('velocity', {}) or {}
            peak_vel = vel_data.get('max', vel_data.get('overall_score', 0)) or 0
        peak_velocities.append(float(peak_vel))
    avg_peak = np.mean(peak_velocities) if peak_velocities else 0
    # Normalize assuming upper bound around 6 m/s
    return float(min(100, (avg_peak / 6.0) * 100))


def calculate_distance_control(fencer_data: List[Dict]) -> float:
    """Calculate distance control score based on good attack distance rate (0-100).
    A touch is good distance if any of its attacking intervals are marked good_attack_distance,
    else fallback to attack_distance proximity if available.
    """
    if not fencer_data:
        return 0
    total = len(fencer_data)
    good_count = 0
    for f in fencer_data:
        # Prefer explicit boolean computed from interval flags
        if f.get('good_attack_distance') is True:
            good_count += 1
        else:
            # Fallback to proximity check if only aggregate available
            attack_distance = f.get('attack_distance')
            if isinstance(attack_distance, (int, float)) and attack_distance > 0:
                if abs(attack_distance - 2.0) <= 0.3:
                    good_count += 1
    return float((good_count / total) * 100) if total > 0 else 0


def calculate_timing_precision(fencer_data: List[Dict]) -> float:
    """Calculate timing precision score (0-100). Uses attack acceleration if available; otherwise arm/lunge timing."""
    if not fencer_data:
        return 0
    # If attack_acceleration is present, use it as a proxy for timing precision
    acc_values = [float(f.get('attack_acceleration', 0) or 0) for f in fencer_data if 'attack_acceleration' in f]
    if acc_values:
        avg_acc = np.mean(acc_values)
        # Normalize assuming ~20 m/s² as an upper bound
        return float(min(100, (avg_acc / 20.0) * 100))
    # Fallback to arm/lunge timing if acceleration not present
    timing_scores = []
    for f in fencer_data:
        arm_timing = f.get('arm_timing', {}) or {}
        lunge_timing = f.get('lunge', {}) or {}
        arm_score = float(arm_timing.get('timing_score', 0) or 0)
        lunge_score = float(lunge_timing.get('timing_score', 0) or 0)
        avg_score = (arm_score + lunge_score) / 2.0
        timing_scores.append(avg_score * 100)
    return float(np.mean(timing_scores)) if timing_scores else 0


def calculate_tactical_variety(fencer_data: List[Dict]) -> float:
    """Calculate tactical variety based on attack types."""
    if not fencer_data:
        return 0
    attack_types = set()
    for f in fencer_data:
        attack_type = f.get('attack_type_last_specific', '无攻击')
        if attack_type != '无攻击':
            attack_types.add(attack_type)
    # Score based on number of different attack types used
    variety_score = min(100, len(attack_types) * 25)  # Up to 4 types = 100%
    return variety_score


def calculate_attack_efficiency(fencer_data: List[Dict]) -> float:
    """Calculate attack efficiency score."""
    if not fencer_data:
        return 0
    # Combine success rate with speed/timing
    success_rate = calculate_attack_success_rate(fencer_data)
    velocity_score = calculate_avg_velocity(fencer_data)
    timing_score = calculate_timing_precision(fencer_data)
    return (success_rate + velocity_score + timing_score) / 3


def calculate_defense_success_rate(fencer_data: List[Dict]) -> float:
    """Calculate defense success rate."""
    if not fencer_data:
        return 0
    wins = sum(1 for f in fencer_data if f.get('is_winner', False))
    return (wins / len(fencer_data)) * 100


def calculate_retreat_quality(fencer_data: List[Dict]) -> float:
    """Calculate retreat quality score."""
    # Simplified - use pause control as proxy
    if not fencer_data:
        return 0
    good_retreats = sum(1 for f in fencer_data if not f.get('pause', {}).get('present', False))
    return (good_retreats / len(fencer_data)) * 100


def calculate_counter_opportunity(fencer_data: List[Dict]) -> float:
    """Calculate counter opportunity utilization score (0-100)."""
    if not fencer_data:
        return 0
    # Prefer explicit counter_taken_rate from defense analysis
    rates = [float(f.get('counter_taken_rate', 0) or 0) for f in fencer_data if 'counter_taken_rate' in f]
    if rates:
        return float(np.mean(rates) * 100)
    # Fallback heuristic: use reaction timing as proxy
    reaction_scores = []
    for f in fencer_data:
        reaction_scores.append(calculate_timing_score([f]))
    return float(np.mean(reaction_scores)) if reaction_scores else 0


def calculate_distance_safety(fencer_data: List[Dict]) -> float:
    """Calculate distance safety score (0-100)."""
    if not fencer_data:
        return 0
    # Prefer maintained_safe_distance from defense analysis
    flags = [1 for f in fencer_data if f.get('maintained_safe_distance', False)]
    if flags:
        return float((sum(flags) / len(fencer_data)) * 100)
    # Fallback: early extension as proxy for managing distance
    safe_distances = sum(1 for f in fencer_data if f.get('arm_timing', {}).get('early_extension', False))
    return float((safe_distances / len(fencer_data)) * 100) if fencer_data else 0


def calculate_reaction_speed(fencer_data: List[Dict]) -> float:
    """Calculate reaction speed score."""
    return calculate_timing_score(fencer_data)


def calculate_positioning_score(fencer_data: List[Dict]) -> float:
    """Calculate positioning score."""
    # Use consistency as proxy for good positioning
    return calculate_consistency_score(fencer_data)