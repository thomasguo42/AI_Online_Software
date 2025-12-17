import logging
import json
import os
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Any, Optional, Tuple
from collections import Counter
import matplotlib
from .touch_visualization import create_performance_radar_chart
# Robust Chinese font fallbacks to ensure CJK rendering
matplotlib.rcParams['font.sans-serif'] = [
    'Noto Sans CJK SC',
    'Noto Sans SC',
    'WenQuanYi Micro Hei',
    'WenQuanYi Zen Hei',
    'Microsoft YaHei',
    'SimHei',
    'DejaVu Sans',
    'sans-serif'
]
matplotlib.rcParams['axes.unicode_minus'] = False

def extract_inbox_bout_with_winner(bout_data: Dict, fps: int, bout_classifications: List[Dict]) -> Optional[Dict]:
    """
    Extract detailed In-Box analysis from a single bout including winner information.
    
    Args:
        bout_data: Single bout data from match_analysis.json
        fps: Frames per second
        bout_classifications: List of bout classifications with winner info
    
    Returns:
        Dictionary with In-Box analysis for both fencers and winner, or None if not In-Box
    """
    frame_range = bout_data.get('frame_range', [0, 60])
    total_frames = frame_range[1] - frame_range[0] + 1
    
    # Check if this is an In-Box bout (< 60 frames)
    if total_frames >= 60:
        return None
    
    # Ensure fps is valid
    if fps is None or fps <= 0:
        fps = 30  # Default fallback
    
    match_idx = bout_data.get('match_idx')
    bout_duration_s = total_frames / fps
    
    # Find winner information from bout classifications
    winner_side = None
    for classification in bout_classifications or []:
        if classification.get('match_idx') == match_idx:
            # Classification comes from classify_all_bouts which has direct 'winner' field
            winner_side = classification.get('winner', 'undetermined')
            break
    
    result = {
        'meta': {
            'fps': fps,
            'total_frames': total_frames,
            'bout_duration_s': bout_duration_s,
            'frame_range': frame_range,
            'match_idx': match_idx,
            'winner_side': winner_side
        },
        'left_fencer': {},
        'right_fencer': {}
    }
    
    # Process both fencers
    for side in ['left', 'right']:
        fencer_key = f'{side}_fencer'
        data_key = f'{side}_data'
        
        if data_key not in bout_data:
            continue
            
        fencer_data = bout_data[data_key]
        
        # Extract velocity/acceleration metrics directly from match analysis JSON
        # Use the primary fields from match analysis JSON structure
        overall_velocity = fencer_data.get('velocity', 0.0)
        overall_acceleration = fencer_data.get('acceleration', 0.0)
        attacking_velocity = fencer_data.get('attacking_velocity', 0.0)
        attacking_acceleration = fencer_data.get('attacking_acceleration', 0.0)

        velocity_stats = {
            'mean': overall_velocity,
            'max': attacking_velocity,  # Use attacking velocity as max
            'overall_score': overall_velocity
        }
        
        # Extract acceleration metrics
        acceleration_stats = {
            'mean': overall_acceleration,
            'max': attacking_acceleration,  # Use attacking acceleration as max
            'overall_score': overall_acceleration
        }
        
        # Extract pause information (prefer frame intervals, fallback to seconds)
        pause_intervals_frames = fencer_data.get('pause', []) or []
        pause_intervals_sec = fencer_data.get('pause_sec', []) or []
        if pause_intervals_sec:
            total_pause_duration = sum([(end - start) for start, end in pause_intervals_sec])
            pause_present = True
            pause_count = len(pause_intervals_sec)
        elif pause_intervals_frames:
            total_pause_duration = sum([(end - start) / fps for start, end in pause_intervals_frames])
            pause_present = True
            pause_count = len(pause_intervals_frames)
        else:
            total_pause_duration = 0.0
            pause_present = False
            pause_count = 0

        pause_stats = {
            'present': pause_present,
            'total_seconds': total_pause_duration,
            'share': total_pause_duration / bout_duration_s if bout_duration_s > 0 else 0,
            'count': pause_count
        }
        
        # Extract lunge information
        has_launch = fencer_data.get('has_launch', False)
        launches = fencer_data.get('launches', [])
        lunge_stats = {
            'present': has_launch,
            'launch_time_s': None,
            'peak_velocity': 0.0,
            'timing_score': 0.0
        }
        
        if has_launch and launches:
            first_launch = launches[0]
            launch_time = first_launch.get('start_frame', 0) / fps
            lunge_stats.update({
                'launch_time_s': launch_time,
                'peak_velocity': first_launch.get('front_foot_max_velocity', 0.0),
                'timing_score': 1.0 - (launch_time / bout_duration_s) if bout_duration_s > 0 else 0
            })
        
        # Extract arm extension information from match analysis JSON structure
        # arm_extensions is an array of [start_frame, end_frame] pairs
        arm_extensions = fencer_data.get('arm_extensions', [])
        arm_extensions_sec = fencer_data.get('arm_extensions_sec', [])
        
        # Calculate total extension duration
        if arm_extensions_sec:
            # Use seconds data if available
            extension_duration = sum([(end - start) for start, end in arm_extensions_sec])
        elif arm_extensions:
            # Convert frame data to seconds
            extension_duration = sum([(end - start) / fps for start, end in arm_extensions])
        else:
            extension_duration = 0.0
            
        arm_timing_stats = {
            'present': len(arm_extensions) > 0 or len(arm_extensions_sec) > 0,
            'duration_seconds': extension_duration,
            'ratio': extension_duration / bout_duration_s if bout_duration_s > 0 else 0,
            'early_extension': False,
            'timing_score': 0.0
        }
        
        # Check for early arm extension (within first 0.5 seconds)
        if arm_extensions_sec:
            first_extension_start = arm_extensions_sec[0][0] if arm_extensions_sec else float('inf')
        elif arm_extensions:
            first_extension_start = arm_extensions[0][0] / fps if arm_extensions else float('inf')
        else:
            first_extension_start = float('inf')
            
        if first_extension_start < float('inf'):
            arm_timing_stats['early_extension'] = first_extension_start < 0.5
            arm_timing_stats['timing_score'] = max(0, 1.0 - (first_extension_start / bout_duration_s))
        
        # Extract initial step information
        first_step = fencer_data.get('first_step', {})
        initial_step_stats = {
            'onset_time_s': first_step.get('init_time', 0.0),
            'velocity': first_step.get('velocity', 0.0),
            'acceleration': first_step.get('acceleration', 0.0),
            'momentum': first_step.get('momentum', 0.0),
            'timing_score': 0.0
        }
        
        # Calculate timing score (earlier is better)
        init_time = first_step.get('init_time', 0.0)
        if bout_duration_s > 0:
            initial_step_stats['timing_score'] = max(0, 1.0 - (init_time / bout_duration_s))
        
        # Extract last attack type from interval analyses and derive specific classification
        attack_type_last: Optional[str] = None
        attack_type_last_specific: Optional[str] = None
        interval_analysis = fencer_data.get('interval_analysis', {}) or {}
        advance_analyses = interval_analysis.get('advance_analyses', []) or []

        def classify_specific_attack(attack_info: Dict[str, Any], has_launch_direct: bool) -> str:
            """Map raw attack_info into a more specific category using correct launch data."""
            num_ext = attack_info.get('num_extensions', 0) or 0
            raw_type = attack_info.get('attack_type', '') or ''
            
            # Use the correct launch data from match analysis JSON
            # instead of the faulty num_launches from bout analysis
            if has_launch_direct:
                if num_ext and num_ext > 1:
                    return '佯攻弓步'  # Feint Lunge
                return '直接弓步'      # Direct Lunge
            # No lunge variants
            if raw_type == 'simple_attack' or (num_ext >= 1 and not has_launch_direct):
                return '无弓步攻击'
            if raw_type == 'compound_attack':
                return '复合攻击（无弓步）'
            if raw_type == 'holding_attack':
                return '保持攻击'
            if raw_type == 'simple_preparation':
                return '准备动作'
            return '无攻击'

        # Use only the last advance interval that has an attack
        for advance in (advance_analyses or []):
            attack_info = advance.get('attack_info', {}) or {}
            if attack_info.get('has_attack'):
                attack_type_last = attack_info.get('attack_type') or attack_type_last
                # Pass the correct launch data from match analysis JSON
                attack_type_last_specific = classify_specific_attack(attack_info, has_launch)

        # Determine if this fencer won
        is_winner = (winner_side == side) if winner_side in ['left', 'right'] else None
        
        # Compile all stats for this fencer
        result[fencer_key] = {
            'is_winner': is_winner,
            'has_launch': bool(has_launch),
            'lunge_present': bool(has_launch),
            'velocity': velocity_stats,
            'acceleration': acceleration_stats,
            'pause': pause_stats,
            'lunge': lunge_stats,
            'arm_timing': arm_timing_stats,
            'initial_step': initial_step_stats,
            'attack_type_last': attack_type_last or '无攻击',
            'attack_type_last_specific': attack_type_last_specific or ('无攻击' if not attack_type_last else attack_type_last)
        }
    
    return result

def create_comprehensive_inbox_charts(inbox_bouts: List[Dict], output_dir: str) -> Dict[str, str]:
    """
    Create comprehensive In-Box analysis charts matching user specifications:
    1. Performance Dashboard (KPI Cards)
    2. Action-Outcome Correlation Tables
    3. Timing Coordination Scatter Plots
    4. Velocity/Acceleration Box Plots
    
    Args:
        inbox_bouts: List of processed In-Box bout data with winner information
        output_dir: Directory to save charts
    
    Returns:
        Dictionary mapping chart names to file paths
    """
    os.makedirs(output_dir, exist_ok=True)
    chart_paths = {}
    
    if not inbox_bouts:
        logging.warning("No In-Box data available for charting")
        return chart_paths
    
    # Separate data by fencer and win/loss
    left_wins = []
    left_losses = []
    right_wins = []
    right_losses = []
    all_left = []
    all_right = []
    
    for bout in inbox_bouts:
        left_fencer = bout.get('left_fencer', {})
        right_fencer = bout.get('right_fencer', {})
        
        all_left.append(left_fencer)
        all_right.append(right_fencer)
        
        if left_fencer.get('is_winner') == True:
            left_wins.append(left_fencer)
        elif left_fencer.get('is_winner') == False:
            left_losses.append(left_fencer)
        
        if right_fencer.get('is_winner') == True:
            right_wins.append(right_fencer)
        elif right_fencer.get('is_winner') == False:
            right_losses.append(right_fencer)
    
    # 1. 对攻表现仪表板 - Numeric KPI Comparison
    try:
        fig, ax = plt.subplots(1, 1, figsize=(18, 12))
        fig.suptitle('对攻表现仪表板 - 左右剑手 KPI 对比', fontsize=20, fontweight='bold')
        
        # Calculate KPIs for both fencers
        def calculate_kpis(all_fencer, wins, losses, fencer_name):
            if not all_fencer:
                return {}
            
            # Initiative Rate (% with lunge) — use has_launch flag if available, else lunge.present
            initiative_count = sum([
                1 for f in all_fencer
                if (
                    f.get('lunge', {}).get('present', False)
                    or f.get('lunge_present', False)
                    or f.get('has_launch', False)
                )
            ])
            initiative_rate = (initiative_count / len(all_fencer) * 100) if all_fencer else 0
            
            # Hesitation Rate (% with pause)
            hesitation_count = sum([1 for f in all_fencer if f.get('pause', {}).get('present', False)])
            hesitation_rate = (hesitation_count / len(all_fencer) * 100) if all_fencer else 0
            
            # Average Velocity - check both possible field names
            velocities = []
            for f in all_fencer:
                vel_data = f.get('velocity', {})
                vel = vel_data.get('mean', vel_data.get('overall_score', 0))
                velocities.append(vel)
            avg_velocity = np.mean(velocities) if velocities else 0
            
            # Average Acceleration - check both possible field names
            accelerations = []
            for f in all_fencer:
                acc_data = f.get('acceleration', {})
                acc = acc_data.get('mean', acc_data.get('overall_score', 0))
                accelerations.append(acc)
            avg_acceleration = np.mean(accelerations) if accelerations else 0
            
            # Win Rate
            total_decided = len(wins) + len(losses)
            win_rate = (len(wins) / total_decided * 100) if total_decided > 0 else 0
            
            # Early Arm Extension Rate
            early_arm_count = sum([1 for f in all_fencer if f.get('arm_timing', {}).get('early_extension', False)])
            early_arm_rate = (early_arm_count / len(all_fencer) * 100) if all_fencer else 0
            
            return {
                'name': fencer_name,
                'initiative_rate': initiative_rate,
                'hesitation_rate': hesitation_rate,
                'avg_velocity': avg_velocity,
                'avg_acceleration': avg_acceleration,
                'win_rate': win_rate,
                'early_arm_rate': early_arm_rate,
                'total_bouts': len(all_fencer),
                'wins': len(wins),
                'losses': len(losses)
            }
        
        left_kpis = calculate_kpis(all_left, left_wins, left_losses, '左剑手')
        right_kpis = calculate_kpis(all_right, right_wins, right_losses, '右剑手')
        
        # Create KPI comparison table
        if left_kpis and right_kpis:
            table_data = [
                ['关键指标 (KPI)', '左剑手', '右剑手', '优势方'],
                ['胜率 (%)', f"{left_kpis['win_rate']:.1f}%", f"{right_kpis['win_rate']:.1f}%", 
                 '左剑手' if left_kpis['win_rate'] > right_kpis['win_rate'] else ('右剑手' if right_kpis['win_rate'] > left_kpis['win_rate'] else '平局')],
                ['弓步率 (%)', f"{left_kpis['initiative_rate']:.1f}%", f"{right_kpis['initiative_rate']:.1f}%",
                 '左剑手' if left_kpis['initiative_rate'] > right_kpis['initiative_rate'] else ('右剑手' if right_kpis['initiative_rate'] > left_kpis['initiative_rate'] else '平局')],
                ['停顿率 (%)', f"{left_kpis['hesitation_rate']:.1f}%", f"{right_kpis['hesitation_rate']:.1f}%",
                 '右剑手' if left_kpis['hesitation_rate'] > right_kpis['hesitation_rate'] else ('左剑手' if right_kpis['hesitation_rate'] > left_kpis['hesitation_rate'] else '平局')],  # Lower is better
                ['平均速度 (m/s)', f"{left_kpis['avg_velocity']:.2f}", f"{right_kpis['avg_velocity']:.2f}",
                 '左剑手' if left_kpis['avg_velocity'] > right_kpis['avg_velocity'] else ('右剑手' if right_kpis['avg_velocity'] > left_kpis['avg_velocity'] else '平局')],
                ['平均加速度 (m/s²)', f"{left_kpis['avg_acceleration']:.2f}", f"{right_kpis['avg_acceleration']:.2f}",
                 '左剑手' if left_kpis['avg_acceleration'] > right_kpis['avg_acceleration'] else ('右剑手' if right_kpis['avg_acceleration'] > left_kpis['avg_acceleration'] else '平局')],
                ['提前出剑率 (%)', f"{left_kpis['early_arm_rate']:.1f}%", f"{right_kpis['early_arm_rate']:.1f}%",
                 '左剑手' if left_kpis['early_arm_rate'] > right_kpis['early_arm_rate'] else ('右剑手' if right_kpis['early_arm_rate'] > left_kpis['early_arm_rate'] else '平局')],
                ['总回合数', str(left_kpis['total_bouts']), str(right_kpis['total_bouts']), '-'],
                ['胜利次数', str(left_kpis['wins']), str(right_kpis['wins']), '-'],
                ['失败次数', str(left_kpis['losses']), str(right_kpis['losses']), '-']
            ]
            
            # Create table
            table = ax.table(cellText=table_data, loc='center', cellLoc='center')
            table.auto_set_font_size(False)
            table.set_fontsize(12)
            table.scale(1, 2.5)
            
            # Style the table
            # Header row
            for i in range(len(table_data[0])):
                table[(0, i)].set_facecolor('#2E86AB')
                table[(0, i)].set_text_props(weight='bold', color='white')
            
            # Color code advantage column
            for row in range(1, len(table_data)):
                advantage = table_data[row][3]
                if advantage == '左剑手':
                    table[(row, 3)].set_facecolor('#E7F3E7')  # Light green
                elif advantage == '右剑手':
                    table[(row, 3)].set_facecolor('#E7E7F3')  # Light blue
                else:
                    table[(row, 3)].set_facecolor('#F5F5F5')  # Light gray
            
            ax.set_title('对攻关键表现指标对比')
            ax.axis('off')
        
        plt.tight_layout()
        dashboard_path = os.path.join(output_dir, 'inbox_performance_dashboard.png')
        plt.savefig(dashboard_path, dpi=300, bbox_inches='tight')
        plt.close()
        chart_paths['performance_dashboard'] = dashboard_path
        
    except Exception as e:
        logging.error(f"Error creating performance dashboard: {e}")
    
    # 2. 动作-结果 关联表（对攻）
    try:
        fig, axes = plt.subplots(1, 2, figsize=(22, 12))
        fig.suptitle('动作结果关联分析', fontsize=20, fontweight='bold')
        
        for fencer_idx, (all_fencer, wins, losses, fencer_name) in enumerate([
            (all_left, left_wins, left_losses, '左剑手'),
            (all_right, right_wins, right_losses, '右剑手')
        ]):
            if not all_fencer:
                continue
                
            ax = axes[fencer_idx]
            
            # Define action characteristics to analyze
            characteristics = [
                ('弓步', lambda f, opp: f.get('lunge', {}).get('present', False)),
                ('无弓步', lambda f, opp: not f.get('lunge', {}).get('present', False)),
                ('检测到停顿', lambda f, opp: f.get('pause', {}).get('present', False)),
                ('无停顿', lambda f, opp: not f.get('pause', {}).get('present', False)),
                ('提前出剑', lambda f, opp: f.get('arm_timing', {}).get('early_extension', False)),
                ('延迟出剑', lambda f, opp: not f.get('arm_timing', {}).get('early_extension', False)),
                ('有出剑', lambda f, opp: f.get('arm_timing', {}).get('present', False)),
                ('无出剑', lambda f, opp: not f.get('arm_timing', {}).get('present', False)),
                ('速度优势', lambda f, opp: f.get('velocity', {}).get('mean', 0) > opp.get('velocity', {}).get('mean', 0)),
                ('速度劣势', lambda f, opp: f.get('velocity', {}).get('mean', 0) <= opp.get('velocity', {}).get('mean', 0)),
                ('加速度优势', lambda f, opp: f.get('acceleration', {}).get('mean', 0) > opp.get('acceleration', {}).get('mean', 0)),
                ('加速度劣势', lambda f, opp: f.get('acceleration', {}).get('mean', 0) <= opp.get('acceleration', {}).get('mean', 0)),
                ('反应更快', lambda f, opp: f.get('initial_step', {}).get('onset_time_s', float('inf')) < opp.get('initial_step', {}).get('onset_time_s', float('inf'))),
                ('反应较慢', lambda f, opp: f.get('initial_step', {}).get('onset_time_s', float('inf')) >= opp.get('initial_step', {}).get('onset_time_s', float('inf')))
            ]
            
            table_data = []
            table_data.append(['动作特征', '我方得分', '对方得分', '成功率'])
            
            # Build opponent pairs for each bout
            fencer_opponent_pairs = []
            for bout in inbox_bouts:
                left_fencer = bout.get('left_fencer', {})
                right_fencer = bout.get('right_fencer', {})
                
                if fencer_idx == 0:  # Left fencer
                    if left_fencer in all_fencer:
                        fencer_opponent_pairs.append((left_fencer, right_fencer))
                else:  # Right fencer
                    if right_fencer in all_fencer:
                        fencer_opponent_pairs.append((right_fencer, left_fencer))
            
            for char_name, char_func in characteristics:
                # Count wins with this characteristic
                my_scores = 0
                for fencer, opponent in fencer_opponent_pairs:
                    if fencer in wins and char_func(fencer, opponent):
                        my_scores += 1
                
                # Count losses with this characteristic 
                opp_scores = 0
                for fencer, opponent in fencer_opponent_pairs:
                    if fencer in losses and char_func(fencer, opponent):
                        opp_scores += 1
                
                total = my_scores + opp_scores
                success_rate = (my_scores / total * 100) if total > 0 else 0
                
                table_data.append([
                    char_name,
                    str(my_scores),
                    str(opp_scores),
                    f'{success_rate:.0f}%'
                ])
            
            # Create table
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
                    color = '#E7F3E7'  # Light green
                elif success_rate >= 40:
                    color = '#FFF2E7'  # Light orange
                else:
                    color = '#FFE7E7'  # Light red
                
                for col in range(len(table_data[0])):
                    table[(row, col)].set_facecolor(color)
            
            ax.set_title(f'{fencer_name} - 动作效果分析')
            ax.axis('off')
        
        plt.tight_layout()
        correlation_table_path = os.path.join(output_dir, 'inbox_action_outcome_correlation.png')
        plt.savefig(correlation_table_path, dpi=300, bbox_inches='tight')
        plt.close()
        chart_paths['action_outcome_correlation'] = correlation_table_path
        
    except Exception as e:
        logging.error(f"Error creating action-outcome correlation: {e}")
    
    # 2b. 进攻类型成败统计（对攻）
    try:
        fig, axes = plt.subplots(1, 2, figsize=(22, 12))
        fig.suptitle('进攻类型成败统计', fontsize=20, fontweight='bold')

        label_map = {
            'simple_attack': '简单攻击',
            'compound_attack': '复合攻击',
            'holding_attack': '保持攻击',
            'preparation_attack': '准备攻击',
            'simple_preparation': '简单准备',
            'no_attacks': '无攻击',
        }

        for idx, (all_fencer, fencer_name) in enumerate([
            (all_left, '左剑手'),
            (all_right, '右剑手')
        ]):
            ax = axes[idx]
            win_counts = {}
            loss_counts = {}

            for f in all_fencer:
                is_win = f.get('is_winner') is True
                t = f.get('attack_type_last', '无攻击')
                if is_win:
                    win_counts[t] = win_counts.get(t, 0) + 1
                else:
                    loss_counts[t] = loss_counts.get(t, 0) + 1

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
            ax.set_xticklabels([label_map.get(t, t) for t in all_types], rotation=30, ha='right')
            ax.set_ylabel('次数')
            ax.set_title(f'{fencer_name} - 攻击类型')
            ax.legend()
            ax.grid(True, axis='y', alpha=0.3)

        plt.tight_layout()
        attack_types_path = os.path.join(output_dir, 'inbox_attack_type_outcomes.png')
        plt.savefig(attack_types_path, dpi=300, bbox_inches='tight')
        plt.close()
        chart_paths['attack_type_outcomes'] = attack_types_path

    except Exception as e:
        logging.error(f"Error creating attack type outcomes chart: {e}")

    # 2c. 细分类进攻类型成败统计（对攻）
    try:
        fig, axes = plt.subplots(1, 2, figsize=(22, 12))
        fig.suptitle('细分类进攻类型成败统计', fontsize=20, fontweight='bold')

        for idx, (all_fencer, fencer_name) in enumerate([
            (all_left, '左剑手'),
            (all_right, '右剑手')
        ]):
            ax = axes[idx]
            win_counts = {}
            loss_counts = {}

            for f in all_fencer:
                is_win = f.get('is_winner') is True
                t = f.get('attack_type_last_specific', '无攻击')
                if is_win:
                    win_counts[t] = win_counts.get(t, 0) + 1
                else:
                    loss_counts[t] = loss_counts.get(t, 0) + 1

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
            ax.set_xticklabels([label_map.get(t, t) for t in all_types], rotation=30, ha='right')
            ax.set_ylabel('次数')
            ax.set_title(f'{fencer_name} - 细分类攻击类型')
            ax.legend()
            ax.grid(True, axis='y', alpha=0.3)

        plt.tight_layout()
        attack_types_specific_path = os.path.join(output_dir, 'inbox_attack_type_outcomes_specific.png')
        plt.savefig(attack_types_specific_path, dpi=300, bbox_inches='tight')
        plt.close()
        chart_paths['attack_type_outcomes_specific'] = attack_types_specific_path

    except Exception as e:
        logging.error(f"Error creating specific attack type outcomes chart: {e}")
    
    # Removed timing coordination and velocity/acceleration charts per user request
    
    return chart_paths

def process_inbox_bouts_with_winners(analysis_dir: str, bout_classifications: List[Dict]) -> Dict[str, Any]:
    """
    Process all match analysis files and extract In-Box bout details with winner information.
    
    Args:
        analysis_dir: Directory containing match_analysis JSON files
        bout_classifications: List of bout classifications with winner info
    
    Returns:
        Dictionary with processed In-Box data including winner correlations
    """
    inbox_bouts = []
    
    if not os.path.exists(analysis_dir):
        logging.warning(f"Analysis directory not found: {analysis_dir}")
        return {'bouts': [], 'summary': {}}
    
    # Process all match analysis files
    # Build per-bout per-side inbox flags from classifications
    inbox_lookup = {c.get('match_idx'): (
        c.get('left_category') == 'in_box',
        c.get('right_category') == 'in_box'
    ) for c in (bout_classifications or []) if c.get('match_idx') is not None}
    for filename in os.listdir(analysis_dir):
        if not filename.endswith('_analysis.json'):
            continue
            
        filepath = os.path.join(analysis_dir, filename)
        
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                raw = f.read()
            # Replace invalid JSON tokens
            sanitized = (
                raw.replace('Infinity', 'null')
                   .replace('-Infinity', 'null')
                   .replace('NaN', 'null')
            )
            bout_data = json.loads(sanitized)
            
            fps = bout_data.get('fps', 30)
            inbox_details = extract_inbox_bout_with_winner(bout_data, fps, bout_classifications)
            
            if inbox_details:
                inbox_details['filename'] = filename
                # attach per-side flags for filtering in charts if needed
                flags = inbox_lookup.get(bout_data.get('match_idx'), (False, False))
                inbox_details['left_is_inbox'] = bool(flags[0])
                inbox_details['right_is_inbox'] = bool(flags[1])
                inbox_bouts.append(inbox_details)
                
        except Exception as e:
            logging.error(f"Error processing {filepath}: {e}")
            continue
    
    # Generate comprehensive charts
    chart_output_dir = os.path.join(os.path.dirname(analysis_dir), 'fencer_analysis', 'touch_category_charts', 'inbox')
    chart_paths = create_comprehensive_inbox_charts(inbox_bouts, chart_output_dir)
    
    # Generate performance radar chart
    try:
        radar_chart_path = create_performance_radar_chart(inbox_bouts, 'in_box', chart_output_dir)
        chart_paths['performance_radar'] = radar_chart_path
        logging.info(f"Generated in-box performance radar chart: {radar_chart_path}")
    except Exception as e:
        logging.error(f"Error generating in-box radar chart: {e}")
    
    # Generate summary with win/loss analysis
    summary = generate_comprehensive_inbox_summary(inbox_bouts)
    
    return {
        'bouts': inbox_bouts,
        'summary': summary,
        'chart_paths': chart_paths,
        'total_inbox_bouts': len(inbox_bouts)
    }

def generate_comprehensive_inbox_summary(inbox_bouts: List[Dict]) -> Dict[str, Any]:
    """
    Generate comprehensive summary statistics for In-Box bouts including win/loss analysis.
    
    Args:
        inbox_bouts: List of processed In-Box bout data with winner information
    
    Returns:
        Comprehensive summary statistics dictionary
    """
    if not inbox_bouts:
        return {}
    
    # Separate by fencer and outcome
    left_wins = [bout['left_fencer'] for bout in inbox_bouts if bout['left_fencer'].get('is_winner') == True]
    left_losses = [bout['left_fencer'] for bout in inbox_bouts if bout['left_fencer'].get('is_winner') == False]
    right_wins = [bout['right_fencer'] for bout in inbox_bouts if bout['right_fencer'].get('is_winner') == True]
    right_losses = [bout['right_fencer'] for bout in inbox_bouts if bout['right_fencer'].get('is_winner') == False]
    
    def analyze_parameter_advantage(wins, losses, category, metric):
        """Calculate advantage of wins over losses for a specific parameter."""
        if not wins or not losses:
            return 0.0
        
        win_vals = [f.get(category, {}).get(metric, 0) for f in wins]
        loss_vals = [f.get(category, {}).get(metric, 0) for f in losses]
        
        win_mean = np.mean(win_vals) if win_vals else 0
        loss_mean = np.mean(loss_vals) if loss_vals else 0
        
        return win_mean - loss_mean
    
    def analyze_comparative_advantage(fencer_wins, fencer_losses, opponent_wins, opponent_losses, category, metric):
        """Calculate how often this fencer has advantage over opponent when winning vs losing."""
        if not fencer_wins or not fencer_losses:
            return 0.0
        
        # Count times fencer had advantage when winning
        win_advantages = 0
        for i, fencer_bout in enumerate(fencer_wins):
            if i < len(opponent_wins):  # Ensure we have opponent data
                opp_bout = opponent_wins[i] if len(opponent_wins) > i else {}
                fencer_val = fencer_bout.get(category, {}).get(metric, 0)
                opp_val = opp_bout.get(category, {}).get(metric, 0)
                if fencer_val > opp_val:
                    win_advantages += 1
        
        # Count times fencer had advantage when losing
        loss_advantages = 0 
        for i, fencer_bout in enumerate(fencer_losses):
            if i < len(opponent_losses):  # Ensure we have opponent data
                opp_bout = opponent_losses[i] if len(opponent_losses) > i else {}
                fencer_val = fencer_bout.get(category, {}).get(metric, 0)
                opp_val = opp_bout.get(category, {}).get(metric, 0)
                if fencer_val > opp_val:
                    loss_advantages += 1
        
        # Return difference in advantage rate
        win_advantage_rate = win_advantages / len(fencer_wins) if fencer_wins else 0
        loss_advantage_rate = loss_advantages / len(fencer_losses) if fencer_losses else 0
        
        return win_advantage_rate - loss_advantage_rate
    
    # Analyze parameter advantages
    parameters = [
        ('velocity', 'overall_score'),
        ('acceleration', 'overall_score'),
        ('arm_timing', 'timing_score'),
        ('initial_step', 'timing_score'),
        ('initial_step', 'velocity'),
        ('lunge', 'timing_score')
    ]
    
    left_advantages = {}
    right_advantages = {}
    
    for category, metric in parameters:
        param_name = f"{category}_{metric}"
        left_advantages[param_name] = analyze_parameter_advantage(left_wins, left_losses, category, metric)
        right_advantages[param_name] = analyze_parameter_advantage(right_wins, right_losses, category, metric)
    
    # Add comparative advantages (vs opponent)
    comparative_params = [
        ('velocity', 'mean'),
        ('acceleration', 'mean'), 
        ('initial_step', 'onset_time_s')  # Lower is better for reaction time
    ]
    
    for category, metric in comparative_params:
        param_name = f"vs_opponent_{category}_{metric}"
        if category == 'initial_step':
            # For reaction time, faster (lower) is better, so we flip the comparison
            left_advantages[param_name] = analyze_comparative_advantage(left_losses, left_wins, right_losses, right_wins, category, metric)
            right_advantages[param_name] = analyze_comparative_advantage(right_losses, right_wins, left_losses, left_wins, category, metric)
        else:
            left_advantages[param_name] = analyze_comparative_advantage(left_wins, left_losses, right_losses, right_wins, category, metric)
            right_advantages[param_name] = analyze_comparative_advantage(right_wins, right_losses, left_losses, left_wins, category, metric)
    
    return {
        'total_bouts': len(inbox_bouts),
        'left_fencer': {
            'total_wins': len(left_wins),
            'total_losses': len(left_losses),
            'win_rate': len(left_wins) / (len(left_wins) + len(left_losses)) if (len(left_wins) + len(left_losses)) > 0 else 0,
            'parameter_advantages': left_advantages
        },
        'right_fencer': {
            'total_wins': len(right_wins),
            'total_losses': len(right_losses),
            'win_rate': len(right_wins) / (len(right_wins) + len(right_losses)) if (len(right_wins) + len(right_losses)) > 0 else 0,
            'parameter_advantages': right_advantages
        }
    }