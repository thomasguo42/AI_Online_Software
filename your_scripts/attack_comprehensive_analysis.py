"""
Comprehensive attack analysis module for fencing touch classification.
Generates detailed charts and analysis for attack-type bouts.
"""

import os
import json
import logging
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import font_manager
from typing import Dict, List, Any
from .touch_visualization import create_performance_radar_chart

# Configure matplotlib for Chinese fonts
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans', 'Arial']
plt.rcParams['axes.unicode_minus'] = False

def extract_attack_bout_details(bout_data: Dict, winner_info: str) -> Dict[str, Any]:
    """
    Extract detailed attack analysis from a single bout including winner information.
    
    Args:
        bout_data: Raw bout analysis data
        winner_info: Winner information ('left', 'right', or 'undetermined')
    
    Returns:
        Dictionary with attack analysis for both fencers and winner, or None if not attack bout
    """
    
    def extract_fencer_attack_data(fencer_data: Dict, is_winner: bool) -> Dict[str, Any]:
        """Extract attack-specific metrics for a fencer."""
        
        # Get attack velocity and acceleration directly from match analysis JSON
        # Primary fields from match analysis JSON structure
        attack_velocity = fencer_data.get('attacking_velocity', 0.0)
        attack_acceleration = fencer_data.get('attacking_acceleration', 0.0)
        overall_velocity = fencer_data.get('velocity', 0.0)
        overall_acceleration = fencer_data.get('acceleration', 0.0)
        
        # Use attacking metrics if available, fallback to overall metrics
        if attack_velocity == 0.0 and overall_velocity > 0.0:
            attack_velocity = overall_velocity
        if attack_acceleration == 0.0 and overall_acceleration > 0.0:
            attack_acceleration = overall_acceleration
        
        # Attack distance (use avg_distance from attacking advance intervals)
        # And good_attack_distance if any attacking interval marked good
        interval_analysis = fencer_data.get('interval_analysis', {}) or {}
        advance_analyses = interval_analysis.get('advance_analyses', []) or []
        attacking_distances = []
        interval_good_flags = []
        for adv in advance_analyses:
            info = adv.get('attack_info', {}) or {}
            if info.get('has_attack'):
                # Prefer interval-level avg_distance
                if 'avg_distance' in adv and isinstance(adv.get('avg_distance'), (int, float)):
                    attacking_distances.append(float(adv.get('avg_distance')))
                # Use interval-level good attack distance flag if present
                interval_good_flags.append(bool(adv.get('good_attack_distance')))
        attack_distance = float(np.mean(attacking_distances)) if attacking_distances else 0.0
        # Good distance if any attacking interval is good
        good_attack_distance = any(interval_good_flags)
        
        # Dangerous close frames (fallback using front_foot_x if available)
        front_foot_x = fencer_data.get('front_foot_x', [])
        dangerous_close_frames = sum(1 for x in front_foot_x if x < 1.5) if front_foot_x else 0
        
        # Lunge presence
        has_lunge = fencer_data.get('has_launch', False) or bool(fencer_data.get('launch_frame'))
        
        # Arm extension presence and duration
        # arm_extensions is an array of [start_frame, end_frame] pairs
        arm_extensions = fencer_data.get('arm_extensions', [])
        arm_extensions_sec = fencer_data.get('arm_extensions_sec', [])
        has_arm_extension = len(arm_extensions) > 0 or len(arm_extensions_sec) > 0
        
        # Calculate arm extension duration
        if arm_extensions_sec:
            arm_extension_duration = sum(end - start for start, end in arm_extensions_sec)
        elif arm_extensions:
            arm_extension_duration = sum(end - start for start, end in arm_extensions) / 30.0
        else:
            arm_extension_duration = 0.0
        
        # Attack type and tempo type: prefer interval_analysis advance_analyses
        # default fallbacks
        attack_type = fencer_data.get('attack_type', 'unknown')
        tempo_type = fencer_data.get('tempo_type', 'unknown')
        # use the last advance with attack_info.has_attack = True
        for adv in advance_analyses:
            info = adv.get('attack_info', {}) or {}
            if info.get('has_attack'):
                raw = info.get('attack_type') or 'simple_attack'
                attack_type = raw
                tempo_type = adv.get('tempo_type') or tempo_type
        # fallback inference from movement patterns if still unknown
        if attack_type == 'unknown':
            advance_intervals = fencer_data.get('advance', [])
            attack_type = 'simple_attack' if len(advance_intervals) == 1 else ('compound_attack' if len(advance_intervals) > 1 else 'no_attack')
        if tempo_type == 'unknown':
            advance_intervals = fencer_data.get('advance', [])
            tempo_changes = len(advance_intervals) - 1 if advance_intervals else 0
            tempo_type = 'steady_tempo' if tempo_changes == 0 else ('variable_tempo' if tempo_changes <= 2 else 'broken_tempo')
        
        return {
            'is_winner': is_winner,
            'attack_velocity': float(attack_velocity) if attack_velocity else 0.0,
            'attack_acceleration': float(attack_acceleration) if attack_acceleration else 0.0,
            'attack_distance': float(attack_distance),
            'good_attack_distance': good_attack_distance,
            'dangerous_close_frames': int(dangerous_close_frames),
            'has_lunge': has_lunge,
            'has_arm_extension': has_arm_extension,
            'arm_extension_duration': float(arm_extension_duration),
            'attack_type': attack_type,
            'tempo_type': tempo_type
        }
    
    # Determine winners
    left_winner = winner_info == 'left'
    right_winner = winner_info == 'right'
    
    result = {
        'meta': {
            'match_idx': bout_data.get('match_idx', 0),
            'winner_side': winner_info,
            'total_frames': bout_data.get('frame_range', [0, 0])[1] - bout_data.get('frame_range', [0, 0])[0]
        },
        'left_fencer': extract_fencer_attack_data(bout_data.get('left_data', {}), left_winner),
        'right_fencer': extract_fencer_attack_data(bout_data.get('right_data', {}), right_winner),
        'filename': f"match_{bout_data.get('match_idx', 0)}_analysis.json"
    }
    
    return result

def create_comprehensive_attack_charts(attack_bouts: List[Dict], output_dir: str) -> Dict[str, str]:
    """
    Create comprehensive attack analysis charts:
    1. Attack Distance Chart (with 2.0m optimal line)
    2. Attack Type Win/Loss Bar Charts
    3. Tempo Type Win/Loss Bar Charts  
    4. Attack Velocity/Acceleration Scatter
    5. Attack Performance KPI Dashboard
    6. Action-Outcome Correlation Table
    
    Args:
        attack_bouts: List of processed attack bout data with winner information
        output_dir: Directory to save charts
    
    Returns:
        Dictionary mapping chart names to file paths
    """
    os.makedirs(output_dir, exist_ok=True)
    chart_paths = {}
    
    if not attack_bouts:
        logging.warning("No attack data available for charting")
        return chart_paths
    
    # Separate data by fencer and win/loss
    left_wins = []
    left_losses = []
    right_wins = []
    right_losses = []
    all_left = []
    all_right = []
    
    for bout in attack_bouts:
        left_fencer = bout.get('left_fencer', {})
        right_fencer = bout.get('right_fencer', {})
        left_is_attack = bout.get('left_is_attack', None)
        right_is_attack = bout.get('right_is_attack', None)

        # Only include a side's data in attack charts if that side was classified as 'attack' for this bout
        if left_is_attack is True:
            all_left.append(left_fencer)
            if left_fencer.get('is_winner') == True:
                left_wins.append(left_fencer)
            elif left_fencer.get('is_winner') == False:
                left_losses.append(left_fencer)

        if right_is_attack is True:
            all_right.append(right_fencer)
            if right_fencer.get('is_winner') == True:
                right_wins.append(right_fencer)
            elif right_fencer.get('is_winner') == False:
                right_losses.append(right_fencer)
    
    # 1. Attack Distance Chart
    try:
        fig, axes = plt.subplots(1, 2, figsize=(18, 8))
        fig.suptitle('进攻距离分析', fontsize=16, fontweight='bold')
        
        for idx, (all_fencer, fencer_name) in enumerate([(all_left, '左剑手'), (all_right, '右剑手')]):
            ax = axes[idx]
            if not all_fencer:
                ax.text(0.5, 0.5, '无数据', ha='center', va='center')
                ax.axis('off')
                continue
                
            # Get distances for each bout
            distances = [f.get('attack_distance', 0) for f in all_fencer if f.get('attack_distance', 0) > 0]
            bout_numbers = list(range(1, len(distances) + 1))
            
            if distances:
                ax.scatter(bout_numbers, distances, alpha=0.7, s=50)
                ax.axhline(y=2.0, color='red', linestyle='--', label='最佳距离 (2.0m)')
                ax.set_xlabel('回合序号')
                ax.set_ylabel('进攻距离 (m)')
                ax.set_title(f'{fencer_name} - 进攻距离趋势')
                ax.legend()
                ax.grid(True, alpha=0.3)
            else:
                ax.text(0.5, 0.5, '无有效距离数据', ha='center', va='center')
                ax.axis('off')
        
        plt.tight_layout()
        distance_path = os.path.join(output_dir, 'attack_distance_analysis.png')
        plt.savefig(distance_path, dpi=300, bbox_inches='tight')
        plt.close()
        chart_paths['attack_distance'] = distance_path
        
    except Exception as e:
        logging.error(f"Error creating attack distance chart: {e}")
    
    # 2. Attack Type Win/Loss Charts
    try:
        fig, axes = plt.subplots(1, 2, figsize=(18, 8))
        fig.suptitle('进攻类型成败统计', fontsize=16, fontweight='bold')
        
        attack_type_labels = {
            'simple_attack': '简单攻击',
            'compound_attack': '复合攻击',
            'holding_attack': '保持攻击',
            'preparation_attack': '准备攻击',
            'simple_preparation': '简单准备',
            'no_attack': '无攻击'
        }
        
        for idx, (all_fencer, fencer_name) in enumerate([(all_left, '左剑手'), (all_right, '右剑手')]):
            ax = axes[idx]
            win_counts = {}
            loss_counts = {}
            
            for f in all_fencer:
                is_win = f.get('is_winner') is True
                attack_type = f.get('attack_type', 'no_attack')
                if is_win:
                    win_counts[attack_type] = win_counts.get(attack_type, 0) + 1
                else:
                    loss_counts[attack_type] = loss_counts.get(attack_type, 0) + 1
            
            all_types = sorted(set(list(win_counts.keys()) + list(loss_counts.keys())))
            if not all_types:
                ax.text(0.5, 0.5, '无数据', ha='center', va='center')
                ax.axis('off')
                continue
            
            x = np.arange(len(all_types))
            win_vals = [win_counts.get(t, 0) for t in all_types]
            loss_vals = [loss_counts.get(t, 0) for t in all_types]
            width = 0.35
            
            ax.bar(x - width/2, win_vals, width, label='胜', color='#6CC24A')
            ax.bar(x + width/2, loss_vals, width, label='负', color='#E57373')
            ax.set_xticks(x)
            ax.set_xticklabels([attack_type_labels.get(t, t) for t in all_types], rotation=30, ha='right')
            ax.set_ylabel('次数')
            ax.set_title(f'{fencer_name} - 进攻类型')
            ax.legend()
            ax.grid(True, axis='y', alpha=0.3)
        
        plt.tight_layout()
        attack_type_path = os.path.join(output_dir, 'attack_type_outcomes.png')
        plt.savefig(attack_type_path, dpi=300, bbox_inches='tight')
        plt.close()
        chart_paths['attack_type_outcomes'] = attack_type_path
        
    except Exception as e:
        logging.error(f"Error creating attack type chart: {e}")
    
    # 3. Tempo Type Win/Loss Charts
    try:
        fig, axes = plt.subplots(1, 2, figsize=(18, 8))
        fig.suptitle('节奏类型成败统计', fontsize=16, fontweight='bold')
        
        tempo_type_labels = {
            'steady_tempo': '稳定节奏',
            'variable_tempo': '变化节奏', 
            'broken_tempo': '破碎节奏'
        }
        
        for idx, (all_fencer, fencer_name) in enumerate([(all_left, '左剑手'), (all_right, '右剑手')]):
            ax = axes[idx]
            win_counts = {}
            loss_counts = {}
            
            for f in all_fencer:
                is_win = f.get('is_winner') is True
                tempo_type = f.get('tempo_type', 'steady_tempo')
                if is_win:
                    win_counts[tempo_type] = win_counts.get(tempo_type, 0) + 1
                else:
                    loss_counts[tempo_type] = loss_counts.get(tempo_type, 0) + 1
            
            all_types = sorted(set(list(win_counts.keys()) + list(loss_counts.keys())))
            if not all_types:
                ax.text(0.5, 0.5, '无数据', ha='center', va='center')
                ax.axis('off')
                continue
            
            x = np.arange(len(all_types))
            win_vals = [win_counts.get(t, 0) for t in all_types]
            loss_vals = [loss_counts.get(t, 0) for t in all_types]
            width = 0.35
            
            ax.bar(x - width/2, win_vals, width, label='胜', color='#6CC24A')
            ax.bar(x + width/2, loss_vals, width, label='负', color='#E57373')
            ax.set_xticks(x)
            ax.set_xticklabels([tempo_type_labels.get(t, t) for t in all_types], rotation=30, ha='right')
            ax.set_ylabel('次数')
            ax.set_title(f'{fencer_name} - 节奏类型')
            ax.legend()
            ax.grid(True, axis='y', alpha=0.3)
        
        plt.tight_layout()
        tempo_type_path = os.path.join(output_dir, 'attack_tempo_outcomes.png')
        plt.savefig(tempo_type_path, dpi=300, bbox_inches='tight')
        plt.close()
        chart_paths['attack_tempo_outcomes'] = tempo_type_path
        
    except Exception as e:
        logging.error(f"Error creating tempo type chart: {e}")
    
    # 4. Attack Velocity/Acceleration Scatter
    try:
        fig, axes = plt.subplots(1, 2, figsize=(18, 8))
        fig.suptitle('进攻速度与加速度分析', fontsize=16, fontweight='bold')
        
        for idx, (all_fencer, wins, losses, fencer_name) in enumerate([
            (all_left, left_wins, left_losses, '左剑手'),
            (all_right, right_wins, right_losses, '右剑手')
        ]):
            ax = axes[idx]
            
            if not all_fencer:
                ax.text(0.5, 0.5, '无数据', ha='center', va='center')
                ax.axis('off')
                continue
            
            # Plot wins and losses with different colors
            for data, label, color in [(wins, '胜利', '#6CC24A'), (losses, '失败', '#E57373')]:
                if data:
                    velocities = [f.get('attack_velocity', 0) for f in data]
                    accelerations = [f.get('attack_acceleration', 0) for f in data]
                    ax.scatter(velocities, accelerations, alpha=0.7, label=label, color=color, s=50)
            
            ax.set_xlabel('进攻速度 (m/s)')
            ax.set_ylabel('进攻加速度 (m/s²)')
            ax.set_title(f'{fencer_name} - 速度加速度分布')
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        velocity_path = os.path.join(output_dir, 'attack_velocity_acceleration.png')
        plt.savefig(velocity_path, dpi=300, bbox_inches='tight')
        plt.close()
        chart_paths['attack_velocity_acceleration'] = velocity_path
        
    except Exception as e:
        logging.error(f"Error creating velocity/acceleration chart: {e}")
    
    # 5. Attack Performance KPI Dashboard
    try:
        fig, ax = plt.subplots(1, 1, figsize=(18, 12))
        fig.suptitle('进攻表现KPI仪表板', fontsize=20, fontweight='bold')
        
        def calculate_attack_kpis(all_fencer, wins, losses, fencer_name):
            if not all_fencer:
                return {}
            
            # Win rate
            total_decided = len(wins) + len(losses)
            win_rate = (len(wins) / total_decided * 100) if total_decided > 0 else 0
            
            # Average attack velocity and acceleration
            avg_velocity = np.mean([f.get('attack_velocity', 0) for f in all_fencer]) if all_fencer else 0
            avg_acceleration = np.mean([f.get('attack_acceleration', 0) for f in all_fencer]) if all_fencer else 0
            
            # Good attack distance rate
            good_distance_count = sum(1 for f in all_fencer if f.get('good_attack_distance', False))
            good_distance_rate = (good_distance_count / len(all_fencer) * 100) if all_fencer else 0
            
            # Lunge rate
            lunge_count = sum(1 for f in all_fencer if f.get('has_lunge', False))
            lunge_rate = (lunge_count / len(all_fencer) * 100) if all_fencer else 0
            
            # Arm extension rate
            arm_ext_count = sum(1 for f in all_fencer if f.get('has_arm_extension', False))
            arm_ext_rate = (arm_ext_count / len(all_fencer) * 100) if all_fencer else 0
            
            # Average dangerous close frames
            avg_dangerous_frames = np.mean([f.get('dangerous_close_frames', 0) for f in all_fencer]) if all_fencer else 0
            
            return {
                'name': fencer_name,
                'win_rate': win_rate,
                'avg_velocity': avg_velocity,
                'avg_acceleration': avg_acceleration,
                'good_distance_rate': good_distance_rate,
                'lunge_rate': lunge_rate,
                'arm_ext_rate': arm_ext_rate,
                'avg_dangerous_frames': avg_dangerous_frames,
                'total_bouts': len(all_fencer),
                'wins': len(wins),
                'losses': len(losses)
            }
        
        left_kpis = calculate_attack_kpis(all_left, left_wins, left_losses, '左剑手')
        right_kpis = calculate_attack_kpis(all_right, right_wins, right_losses, '右剑手')
        
        if left_kpis and right_kpis:
            table_data = [
                ['关键指标 (KPI)', '左剑手', '右剑手', '优势方'],
                ['胜率 (%)', f"{left_kpis['win_rate']:.1f}%", f"{right_kpis['win_rate']:.1f}%", 
                 '左剑手' if left_kpis['win_rate'] > right_kpis['win_rate'] else ('右剑手' if right_kpis['win_rate'] > left_kpis['win_rate'] else '平局')],
                ['平均进攻速度 (m/s)', f"{left_kpis['avg_velocity']:.2f}", f"{right_kpis['avg_velocity']:.2f}",
                 '左剑手' if left_kpis['avg_velocity'] > right_kpis['avg_velocity'] else ('右剑手' if right_kpis['avg_velocity'] > left_kpis['avg_velocity'] else '平局')],
                ['平均进攻加速度 (m/s²)', f"{left_kpis['avg_acceleration']:.2f}", f"{right_kpis['avg_acceleration']:.2f}",
                 '左剑手' if left_kpis['avg_acceleration'] > right_kpis['avg_acceleration'] else ('右剑手' if right_kpis['avg_acceleration'] > left_kpis['avg_acceleration'] else '平局')],
                ['良好距离率 (%)', f"{left_kpis['good_distance_rate']:.1f}%", f"{right_kpis['good_distance_rate']:.1f}%",
                 '左剑手' if left_kpis['good_distance_rate'] > right_kpis['good_distance_rate'] else ('右剑手' if right_kpis['good_distance_rate'] > left_kpis['good_distance_rate'] else '平局')],
                ['弓步率 (%)', f"{left_kpis['lunge_rate']:.1f}%", f"{right_kpis['lunge_rate']:.1f}%",
                 '左剑手' if left_kpis['lunge_rate'] > right_kpis['lunge_rate'] else ('右剑手' if right_kpis['lunge_rate'] > left_kpis['lunge_rate'] else '平局')],
                ['出剑率 (%)', f"{left_kpis['arm_ext_rate']:.1f}%", f"{right_kpis['arm_ext_rate']:.1f}%",
                 '左剑手' if left_kpis['arm_ext_rate'] > right_kpis['arm_ext_rate'] else ('右剑手' if right_kpis['arm_ext_rate'] > left_kpis['arm_ext_rate'] else '平局')],
                ['平均危险近身帧数', f"{left_kpis['avg_dangerous_frames']:.1f}", f"{right_kpis['avg_dangerous_frames']:.1f}",
                 '右剑手' if left_kpis['avg_dangerous_frames'] > right_kpis['avg_dangerous_frames'] else ('左剑手' if right_kpis['avg_dangerous_frames'] > left_kpis['avg_dangerous_frames'] else '平局')],  # Lower is better
                ['总回合数', str(left_kpis['total_bouts']), str(right_kpis['total_bouts']), '-'],
                ['胜利次数', str(left_kpis['wins']), str(right_kpis['wins']), '-'],
                ['失败次数', str(left_kpis['losses']), str(right_kpis['losses']), '-']
            ]
            
            table = ax.table(cellText=table_data, loc='center', cellLoc='center')
            table.auto_set_font_size(False)
            table.set_fontsize(12)
            table.scale(1, 2.5)
            
            # Style the table
            for i in range(len(table_data[0])):
                table[(0, i)].set_facecolor('#2E86AB')
                table[(0, i)].set_text_props(weight='bold', color='white')
            
            # Color code advantage column
            for row in range(1, len(table_data)):
                advantage = table_data[row][3]
                if advantage == '左剑手':
                    table[(row, 3)].set_facecolor('#E7F3E7')
                elif advantage == '右剑手':
                    table[(row, 3)].set_facecolor('#E7E7F3')
                else:
                    table[(row, 3)].set_facecolor('#F5F5F5')
            
            ax.set_title('进攻关键表现指标对比')
            ax.axis('off')
        
        plt.tight_layout()
        dashboard_path = os.path.join(output_dir, 'attack_performance_dashboard.png')
        plt.savefig(dashboard_path, dpi=300, bbox_inches='tight')
        plt.close()
        chart_paths['attack_performance_dashboard'] = dashboard_path
        
    except Exception as e:
        logging.error(f"Error creating attack performance dashboard: {e}")
    
    # 6. Action-Outcome Correlation Table
    try:
        fig, axes = plt.subplots(1, 2, figsize=(22, 12))
        fig.suptitle('进攻动作结果关联分析', fontsize=20, fontweight='bold')
        
        for fencer_idx, (all_fencer, wins, losses, fencer_name) in enumerate([
            (all_left, left_wins, left_losses, '左剑手'),
            (all_right, right_wins, right_losses, '右剑手')
        ]):
            if not all_fencer:
                continue
                
            ax = axes[fencer_idx]
            
            characteristics = [
                ('弓步进攻', lambda f: f.get('has_lunge', False)),
                ('无弓步进攻', lambda f: not f.get('has_lunge', False)),
                ('有手臂伸展', lambda f: f.get('has_arm_extension', False)),
                ('无手臂伸展', lambda f: not f.get('has_arm_extension', False)),
                ('良好攻击距离', lambda f: f.get('good_attack_distance', False)),
                ('距离不佳', lambda f: not f.get('good_attack_distance', False)),
                ('简单攻击', lambda f: f.get('attack_type') == 'simple_attack'),
                ('复合攻击', lambda f: f.get('attack_type') == 'compound_attack'),
                ('稳定节奏', lambda f: f.get('tempo_type') == 'steady_tempo'),
                ('变化节奏', lambda f: f.get('tempo_type') == 'variable_tempo'),
                ('破碎节奏', lambda f: f.get('tempo_type') == 'broken_tempo'),
                ('无危险近身', lambda f: f.get('dangerous_close_frames', 0) == 0),
                ('有危险近身', lambda f: f.get('dangerous_close_frames', 0) > 0)
            ]
            
            table_data = []
            table_data.append(['动作特征', '我方得分', '对方得分', '成功率'])
            
            for char_name, char_func in characteristics:
                my_scores = sum(1 for f in wins if char_func(f))
                opp_scores = sum(1 for f in losses if char_func(f))
                total = my_scores + opp_scores
                success_rate = (my_scores / total * 100) if total > 0 else 0
                
                table_data.append([
                    char_name,
                    str(my_scores),
                    str(opp_scores),
                    f'{success_rate:.0f}%'
                ])
            
            table = ax.table(cellText=table_data, loc='center', cellLoc='center')
            table.auto_set_font_size(False)
            table.set_fontsize(10)
            table.scale(1, 2)
            
            # Style the table
            for i in range(len(table_data[0])):
                table[(0, i)].set_facecolor('#4472C4')
                table[(0, i)].set_text_props(weight='bold', color='white')
            
            # Color code success rates
            for row in range(1, len(table_data)):
                success_rate = float(table_data[row][3].replace('%', ''))
                if success_rate >= 60:
                    color = '#E7F3E7'
                elif success_rate >= 40:
                    color = '#FFF2E7'
                else:
                    color = '#FFE7E7'
                
                for col in range(len(table_data[0])):
                    table[(row, col)].set_facecolor(color)
            
            ax.set_title(f'{fencer_name} - 进攻动作效果分析')
            ax.axis('off')
        
        plt.tight_layout()
        correlation_path = os.path.join(output_dir, 'attack_action_outcome_correlation.png')
        plt.savefig(correlation_path, dpi=300, bbox_inches='tight')
        plt.close()
        chart_paths['attack_action_outcome_correlation'] = correlation_path
        
    except Exception as e:
        logging.error(f"Error creating attack action-outcome correlation: {e}")
    
    # Generate performance radar chart
    try:
        radar_chart_path = create_performance_radar_chart(attack_bouts, 'attack', output_dir)
        chart_paths['performance_radar'] = radar_chart_path
        logging.info(f"Generated attack performance radar chart: {radar_chart_path}")
    except Exception as e:
        logging.error(f"Error generating attack radar chart: {e}")
    
    return chart_paths

def process_attack_bouts_with_winners(analysis_dir: str, bout_classifications: List[Dict]) -> Dict[str, Any]:
    """
    Process all match analysis files and extract attack bout details with winner information.
    
    Args:
        analysis_dir: Directory containing match_analysis JSON files
        bout_classifications: List of bout classifications with winner info
    
    Returns:
        Dictionary with processed attack data including winner correlations
    """
    attack_bouts = []
    
    if not os.path.exists(analysis_dir):
        logging.warning(f"Analysis directory not found: {analysis_dir}")
        return {'bouts': [], 'summary': {}}
    
    # Create winner and category lookup
    winner_lookup = {}
    category_lookup = {}
    for classification in bout_classifications:
        match_idx = classification.get('match_idx')
        if match_idx is not None:
            winner_lookup[match_idx] = classification.get('winner', 'undetermined')
            category_lookup[match_idx] = {
                'left_is_attack': classification.get('left_category') == 'attack',
                'right_is_attack': classification.get('right_category') == 'attack'
            }
    
    # Process analysis files
    for filename in os.listdir(analysis_dir):
        if not filename.endswith('_analysis.json'):
            continue
            
        filepath = os.path.join(analysis_dir, filename)
        
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                raw = f.read()
            sanitized = (
                raw.replace('Infinity', 'null')
                   .replace('-Infinity', 'null')
                   .replace('NaN', 'null')
            )
            data = json.loads(sanitized)
            
            match_idx = data.get('match_idx')
            winner = winner_lookup.get(match_idx, 'undetermined')
            
            # Strict inclusion: only if classified as attack for either side
            is_classified_attack = any(
                c.get('match_idx') == match_idx and (
                    c.get('left_category') == 'attack' or c.get('right_category') == 'attack'
                ) for c in bout_classifications
            )

            if is_classified_attack:
                bout_details = extract_attack_bout_details(data, winner)
                flags = category_lookup.get(match_idx, {})
                bout_details['left_is_attack'] = bool(flags.get('left_is_attack'))
                bout_details['right_is_attack'] = bool(flags.get('right_is_attack'))
                if bout_details:
                    attack_bouts.append(bout_details)
            
        except Exception as e:
            logging.error(f"Error processing {filename}: {e}")
            continue
    
    # Generate summary statistics
    summary = generate_attack_summary_statistics(attack_bouts)
    
    return {
        'bouts': attack_bouts,
        'summary': summary,
        'total_attack_bouts': len(attack_bouts)
    }

def generate_attack_summary_statistics(attack_bouts: List[Dict]) -> Dict[str, Any]:
    """
    Generate comprehensive summary statistics for attack bouts including win/loss analysis.
    
    Args:
        attack_bouts: List of processed attack bout data with winner information
    
    Returns:
        Dictionary with summary statistics and comparisons
    """
    if not attack_bouts:
        return {}
    
    def analyze_side(side_key):
        wins = []
        losses = []
        all_data = []
        
        for bout in attack_bouts:
            fencer_data = bout.get(side_key, {})
            all_data.append(fencer_data)
            
            if fencer_data.get('is_winner') == True:
                wins.append(fencer_data)
            elif fencer_data.get('is_winner') == False:
                losses.append(fencer_data)
        
        total_decided = len(wins) + len(losses)
        win_rate = (len(wins) / total_decided) if total_decided > 0 else 0
        
        # Calculate averages for wins vs losses
        def calc_avg(data_list, key, default=0.0):
            values = [d.get(key, default) for d in data_list if d.get(key) is not None]
            return np.mean(values) if values else default
        
        return {
            'total_wins': len(wins),
            'total_losses': len(losses),
            'win_rate': win_rate,
            'avg_attack_velocity': calc_avg(all_data, 'attack_velocity'),
            'avg_attack_acceleration': calc_avg(all_data, 'attack_acceleration'),
            'avg_attack_distance': calc_avg(all_data, 'attack_distance'),
            'good_distance_rate': sum(1 for d in all_data if d.get('good_attack_distance', False)) / len(all_data) if all_data else 0,
            'lunge_rate': sum(1 for d in all_data if d.get('has_lunge', False)) / len(all_data) if all_data else 0,
            'arm_extension_rate': sum(1 for d in all_data if d.get('has_arm_extension', False)) / len(all_data) if all_data else 0,
            'avg_dangerous_frames': calc_avg(all_data, 'dangerous_close_frames'),
            'win_avg_velocity': calc_avg(wins, 'attack_velocity'),
            'loss_avg_velocity': calc_avg(losses, 'attack_velocity'),
            'win_avg_acceleration': calc_avg(wins, 'attack_acceleration'),
            'loss_avg_acceleration': calc_avg(losses, 'attack_acceleration')
        }
    
    left_summary = analyze_side('left_fencer')
    right_summary = analyze_side('right_fencer')
    
    return {
        'total_bouts': len(attack_bouts),
        'left_fencer': left_summary,
        'right_fencer': right_summary
    }