"""
Comprehensive defense analysis module for fencing touch classification.
Generates detailed charts and analysis for defense-type bouts.
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

def extract_defense_bout_details(bout_data: Dict, winner_info: str) -> Dict[str, Any]:
    """
    Extract detailed defense analysis from a single bout including winner information.
    
    Args:
        bout_data: Raw bout analysis data
        winner_info: Winner information ('left', 'right', or 'undetermined')
    
    Returns:
        Dictionary with defense analysis for both fencers and winner, or None if not defense bout
    """
    
    def extract_fencer_defense_data(fencer_data: Dict, is_winner: bool) -> Dict[str, Any]:
        """Extract defense-specific metrics for a fencer."""
        
        # Defense distance: use avg_distance from retreat_analyses; fallback to pause/retreat mid distance
        front_foot_x = fencer_data.get('front_foot_x', [])
        interval_analysis = fencer_data.get('interval_analysis', {}) or {}
        retreat_level = interval_analysis.get('retreat_analyses', []) or []
        retreat_avg_distances = [float(r.get('avg_distance')) for r in retreat_level if isinstance(r.get('avg_distance'), (int, float))]
        defense_distance = float(np.mean(retreat_avg_distances)) if retreat_avg_distances else (np.mean(front_foot_x) if front_foot_x else 0.0)
        
        # Prefer interval-level flags from retreat_analyses (any True => True). Fallback to heuristics
        # retreat_level already fetched above

        # Maintained safe distance
        maintained_safe_distance = any(bool(r.get('maintained_safe_distance')) for r in retreat_level) if retreat_level else False
        if not maintained_safe_distance and defense_distance > 0:
            maintained_safe_distance = abs(defense_distance - 2.0) <= 0.4

        # Consistent spacing
        consistent_spacing = any(bool(r.get('consistent_spacing')) for r in retreat_level) if retreat_level else False
        if not consistent_spacing and front_foot_x and len(front_foot_x) > 1:
            spacing_variance = np.var(front_foot_x)
            consistent_spacing = spacing_variance < 0.1  # Low variance indicates consistent spacing
        
        # Counter opportunities
        # Use consistent arm extension data from match analysis JSON
        arm_extensions = fencer_data.get('arm_extensions', [])
        arm_extensions_sec = fencer_data.get('arm_extensions_sec', [])
        pause_intervals = fencer_data.get('pause', [])
        pause_intervals_sec = fencer_data.get('pause_sec', [])
        # retreat intervals are nested under movement_data in match analysis JSON
        movement_data = fencer_data.get('movement_data', {}) or {}
        retreat_intervals = movement_data.get('retreat_intervals', []) or []
        
        # Count potential counter opportunities (pauses/retreats followed by extensions)
        counter_opportunities = 0
        counter_taken = 0
        
        # Use second-based intervals for more accurate timing if available
        if pause_intervals_sec and arm_extensions_sec:
            all_defensive_intervals = pause_intervals_sec + retreat_intervals
            counter_opportunities = len(all_defensive_intervals)
            # Count arm extensions that occur after defensive intervals as taken counters
            for def_start, def_end in all_defensive_intervals:
                for ext_start, ext_end in arm_extensions_sec:
                    if ext_start >= def_end:  # Extension starts after defensive interval ends
                        counter_taken += 1
                        break  # Only count one extension per defensive interval
        elif pause_intervals and arm_extensions:
            # Fallback to frame-based calculation
            all_defensive_intervals = pause_intervals + retreat_intervals
            counter_opportunities = len(all_defensive_intervals)
            for def_start, def_end in all_defensive_intervals:
                for ext_start, ext_end in arm_extensions:
                    if ext_start >= def_end:  # Extension starts after defensive interval ends
                        counter_taken += 1
                        break  # Only count one extension per defensive interval
        
        counter_missed = max(0, counter_opportunities - counter_taken)
        counter_taken_rate = (counter_taken / counter_opportunities) if counter_opportunities > 0 else 0
        
        # Defense velocity and acceleration are not meaningful per spec for radar; keep for summary only if needed
        defense_velocity = fencer_data.get('velocity', 0.0)
        defense_acceleration = fencer_data.get('acceleration', 0.0)
        attacking_velocity = fencer_data.get('attacking_velocity', 0.0)
        attacking_acceleration = fencer_data.get('attacking_acceleration', 0.0)
        
        # Defense success indicators
        advance_intervals = fencer_data.get('advance', [])
        advance_frames = sum(end - start + 1 for start, end in advance_intervals)
        total_frames = len(front_foot_x) if front_foot_x else 1
        defense_ratio = 1.0 - (advance_frames / total_frames) if total_frames > 0 else 0.0
        
        return {
            'is_winner': is_winner,
            'defense_distance': float(defense_distance),
            'maintained_safe_distance': maintained_safe_distance,
            'consistent_spacing': consistent_spacing,
            'counter_opportunities': int(counter_opportunities),
            'counter_taken': int(counter_taken),
            'counter_missed': int(counter_missed),
            'counter_taken_rate': float(counter_taken_rate),
            'defense_velocity': float(defense_velocity),
            'defense_acceleration': float(defense_acceleration),
            'attacking_velocity': float(attacking_velocity),
            'attacking_acceleration': float(attacking_acceleration),
            'defense_ratio': float(defense_ratio)
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
        'left_fencer': extract_fencer_defense_data(bout_data.get('left_data', {}), left_winner),
        'right_fencer': extract_fencer_defense_data(bout_data.get('right_data', {}), right_winner),
        'filename': f"match_{bout_data.get('match_idx', 0)}_analysis.json"
    }
    
    return result

def create_comprehensive_defense_charts(defense_bouts: List[Dict], output_dir: str) -> Dict[str, str]:
    """
    Create comprehensive defense analysis charts:
    1. Defense Distance Chart (with 2.0m optimal line)
    2. Counter Opportunities Analysis
    3. Defense Performance KPI Dashboard
    4. Action-Outcome Correlation Table
    5. Spacing Consistency Analysis
    
    Args:
        defense_bouts: List of processed defense bout data with winner information
        output_dir: Directory to save charts
    
    Returns:
        Dictionary mapping chart names to file paths
    """
    os.makedirs(output_dir, exist_ok=True)
    chart_paths = {}
    
    if not defense_bouts:
        logging.warning("No defense data available for charting")
        return chart_paths
    
    # Separate data by fencer and win/loss (only include sides classified as defense)
    left_wins = []
    left_losses = []
    right_wins = []
    right_losses = []
    all_left = []
    all_right = []
    
    for bout in defense_bouts:
        left_fencer = bout.get('left_fencer', {})
        right_fencer = bout.get('right_fencer', {})
        left_is_defense = bool(bout.get('left_is_defense'))
        right_is_defense = bool(bout.get('right_is_defense'))

        if left_is_defense:
            all_left.append(left_fencer)
            if left_fencer.get('is_winner') == True:
                left_wins.append(left_fencer)
            elif left_fencer.get('is_winner') == False:
                left_losses.append(left_fencer)

        if right_is_defense:
            all_right.append(right_fencer)
            if right_fencer.get('is_winner') == True:
                right_wins.append(right_fencer)
            elif right_fencer.get('is_winner') == False:
                right_losses.append(right_fencer)
    
    # 1. Defense Distance Chart
    try:
        fig, axes = plt.subplots(1, 2, figsize=(18, 8))
        fig.suptitle('防守距离分析', fontsize=16, fontweight='bold')
        
        for idx, (all_fencer, fencer_name) in enumerate([(all_left, '左剑手'), (all_right, '右剑手')]):
            ax = axes[idx]
            if not all_fencer:
                ax.text(0.5, 0.5, '无数据', ha='center', va='center')
                ax.axis('off')
                continue
                
            # Get distances for each bout
            distances = [f.get('defense_distance', 0) for f in all_fencer if f.get('defense_distance', 0) > 0]
            bout_numbers = list(range(1, len(distances) + 1))
            
            if distances:
                ax.scatter(bout_numbers, distances, alpha=0.7, s=50)
                ax.axhline(y=2.0, color='red', linestyle='--', label='最佳距离 (2.0m)')
                ax.set_xlabel('回合序号')
                ax.set_ylabel('防守距离 (m)')
                ax.set_title(f'{fencer_name} - 防守距离趋势')
                ax.legend()
                ax.grid(True, alpha=0.3)
            else:
                ax.text(0.5, 0.5, '无有效距离数据', ha='center', va='center')
                ax.axis('off')
        
        plt.tight_layout()
        distance_path = os.path.join(output_dir, 'defense_distance_analysis.png')
        plt.savefig(distance_path, dpi=300, bbox_inches='tight')
        plt.close()
        chart_paths['defense_distance'] = distance_path
        
    except Exception as e:
        logging.error(f"Error creating defense distance chart: {e}")
    
    # 2. Counter Opportunities Analysis
    try:
        fig, axes = plt.subplots(2, 2, figsize=(18, 16))
        fig.suptitle('反击机会分析', fontsize=16, fontweight='bold')
        
        # Counter opportunities bar chart
        ax1 = axes[0, 0]
        fencer_names = ['左剑手', '右剑手']
        total_opportunities = [
            sum(f.get('counter_opportunities', 0) for f in all_left),
            sum(f.get('counter_opportunities', 0) for f in all_right)
        ]
        taken_opportunities = [
            sum(f.get('counter_taken', 0) for f in all_left),
            sum(f.get('counter_taken', 0) for f in all_right)
        ]
        missed_opportunities = [
            sum(f.get('counter_missed', 0) for f in all_left),
            sum(f.get('counter_missed', 0) for f in all_right)
        ]
        
        x = np.arange(len(fencer_names))
        width = 0.35
        
        ax1.bar(x - width/2, taken_opportunities, width, label='已把握', color='#6CC24A')
        ax1.bar(x + width/2, missed_opportunities, width, label='已错失', color='#E57373')
        ax1.set_xticks(x)
        ax1.set_xticklabels(fencer_names)
        ax1.set_ylabel('反击机会数量')
        ax1.set_title('反击机会把握情况')
        ax1.legend()
        ax1.grid(True, axis='y', alpha=0.3)
        
        # Counter success rate
        ax2 = axes[0, 1]
        success_rates = []
        for opportunities, taken in zip(total_opportunities, taken_opportunities):
            rate = (taken / opportunities * 100) if opportunities > 0 else 0
            success_rates.append(rate)
        
        bars = ax2.bar(fencer_names, success_rates, color=['#4A90E2', '#F39C12'])
        ax2.set_ylabel('成功率 (%)')
        ax2.set_title('反击成功率')
        ax2.set_ylim(0, 100)
        
        # Add value labels on bars
        for bar, rate in zip(bars, success_rates):
            height = bar.get_height()
            ax2.annotate(f'{rate:.1f}%',
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 3),
                        textcoords="offset points",
                        ha='center', va='bottom')
        
        # Counter opportunities per bout
        ax3 = axes[1, 0]
        left_opportunities_per_bout = [f.get('counter_opportunities', 0) for f in all_left]
        right_opportunities_per_bout = [f.get('counter_opportunities', 0) for f in all_right]
        
        if left_opportunities_per_bout or right_opportunities_per_bout:
            ax3.boxplot([left_opportunities_per_bout, right_opportunities_per_bout], 
                       labels=fencer_names)
            ax3.set_ylabel('每回合反击机会数')
            ax3.set_title('反击机会分布')
            ax3.grid(True, alpha=0.3)
        else:
            ax3.text(0.5, 0.5, '无数据', ha='center', va='center')
            ax3.axis('off')
        
        # Counter timing analysis
        ax4 = axes[1, 1]
        left_rates = [f.get('counter_taken_rate', 0) * 100 for f in all_left]
        right_rates = [f.get('counter_taken_rate', 0) * 100 for f in all_right]
        
        if left_rates or right_rates:
            ax4.boxplot([left_rates, right_rates], labels=fencer_names)
            ax4.set_ylabel('反击成功率 (%)')
            ax4.set_title('个人反击成功率分布')
            ax4.grid(True, alpha=0.3)
        else:
            ax4.text(0.5, 0.5, '无数据', ha='center', va='center')
            ax4.axis('off')
        
        plt.tight_layout()
        counter_path = os.path.join(output_dir, 'defense_counter_analysis.png')
        plt.savefig(counter_path, dpi=300, bbox_inches='tight')
        plt.close()
        chart_paths['defense_counter_analysis'] = counter_path
        
    except Exception as e:
        logging.error(f"Error creating counter opportunities chart: {e}")
    
    # 3. Defense Performance KPI Dashboard
    try:
        fig, ax = plt.subplots(1, 1, figsize=(18, 12))
        fig.suptitle('防守表现KPI仪表板', fontsize=20, fontweight='bold')
        
        def calculate_defense_kpis(all_fencer, wins, losses, fencer_name):
            if not all_fencer:
                return {}
            
            # Win rate
            total_decided = len(wins) + len(losses)
            win_rate = (len(wins) / total_decided * 100) if total_decided > 0 else 0
            
            # Safe distance maintenance rate
            safe_distance_count = sum(1 for f in all_fencer if f.get('maintained_safe_distance', False))
            safe_distance_rate = (safe_distance_count / len(all_fencer) * 100) if all_fencer else 0
            
            # Consistent spacing rate
            consistent_spacing_count = sum(1 for f in all_fencer if f.get('consistent_spacing', False))
            consistent_spacing_rate = (consistent_spacing_count / len(all_fencer) * 100) if all_fencer else 0
            
            # Counter opportunity stats
            total_opportunities = sum(f.get('counter_opportunities', 0) for f in all_fencer)
            total_taken = sum(f.get('counter_taken', 0) for f in all_fencer)
            counter_success_rate = (total_taken / total_opportunities * 100) if total_opportunities > 0 else 0
            
            # Average defense metrics
            avg_defense_distance = np.mean([f.get('defense_distance', 0) for f in all_fencer]) if all_fencer else 0
            avg_defense_ratio = np.mean([f.get('defense_ratio', 0) for f in all_fencer]) if all_fencer else 0
            
            return {
                'name': fencer_name,
                'win_rate': win_rate,
                'safe_distance_rate': safe_distance_rate,
                'consistent_spacing_rate': consistent_spacing_rate,
                'counter_success_rate': counter_success_rate,
                'avg_defense_distance': avg_defense_distance,
                'avg_defense_ratio': avg_defense_ratio,
                'total_opportunities': total_opportunities,
                'total_taken': total_taken,
                'total_bouts': len(all_fencer),
                'wins': len(wins),
                'losses': len(losses)
            }
        
        left_kpis = calculate_defense_kpis(all_left, left_wins, left_losses, '左剑手')
        right_kpis = calculate_defense_kpis(all_right, right_wins, right_losses, '右剑手')
        
        if left_kpis and right_kpis:
            table_data = [
                ['关键指标 (KPI)', '左剑手', '右剑手', '优势方'],
                ['胜率 (%)', f"{left_kpis['win_rate']:.1f}%", f"{right_kpis['win_rate']:.1f}%", 
                 '左剑手' if left_kpis['win_rate'] > right_kpis['win_rate'] else ('右剑手' if right_kpis['win_rate'] > left_kpis['win_rate'] else '平局')],
                ['安全距离维持率 (%)', f"{left_kpis['safe_distance_rate']:.1f}%", f"{right_kpis['safe_distance_rate']:.1f}%",
                 '左剑手' if left_kpis['safe_distance_rate'] > right_kpis['safe_distance_rate'] else ('右剑手' if right_kpis['safe_distance_rate'] > left_kpis['safe_distance_rate'] else '平局')],
                ['间距一致性率 (%)', f"{left_kpis['consistent_spacing_rate']:.1f}%", f"{right_kpis['consistent_spacing_rate']:.1f}%",
                 '左剑手' if left_kpis['consistent_spacing_rate'] > right_kpis['consistent_spacing_rate'] else ('右剑手' if right_kpis['consistent_spacing_rate'] > left_kpis['consistent_spacing_rate'] else '平局')],
                ['反击成功率 (%)', f"{left_kpis['counter_success_rate']:.1f}%", f"{right_kpis['counter_success_rate']:.1f}%",
                 '左剑手' if left_kpis['counter_success_rate'] > right_kpis['counter_success_rate'] else ('右剑手' if right_kpis['counter_success_rate'] > left_kpis['counter_success_rate'] else '平局')],
                ['平均防守距离 (m)', f"{left_kpis['avg_defense_distance']:.2f}", f"{right_kpis['avg_defense_distance']:.2f}",
                 '左剑手' if abs(left_kpis['avg_defense_distance'] - 2.0) < abs(right_kpis['avg_defense_distance'] - 2.0) else ('右剑手' if abs(right_kpis['avg_defense_distance'] - 2.0) < abs(left_kpis['avg_defense_distance'] - 2.0) else '平局')],
                ['防守比率', f"{left_kpis['avg_defense_ratio']:.2f}", f"{right_kpis['avg_defense_ratio']:.2f}",
                 '左剑手' if left_kpis['avg_defense_ratio'] > right_kpis['avg_defense_ratio'] else ('右剑手' if right_kpis['avg_defense_ratio'] > left_kpis['avg_defense_ratio'] else '平局')],
                ['反击机会总数', str(left_kpis['total_opportunities']), str(right_kpis['total_opportunities']), '-'],
                ['反击把握总数', str(left_kpis['total_taken']), str(right_kpis['total_taken']), '-'],
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
            
            ax.set_title('防守关键表现指标对比')
            ax.axis('off')
        
        plt.tight_layout()
        dashboard_path = os.path.join(output_dir, 'defense_performance_dashboard.png')
        plt.savefig(dashboard_path, dpi=300, bbox_inches='tight')
        plt.close()
        chart_paths['defense_performance_dashboard'] = dashboard_path
        
    except Exception as e:
        logging.error(f"Error creating defense performance dashboard: {e}")
    
    # 4. Action-Outcome Correlation Table
    try:
        fig, axes = plt.subplots(1, 2, figsize=(22, 12))
        fig.suptitle('防守动作结果关联分析', fontsize=20, fontweight='bold')
        
        for fencer_idx, (all_fencer, wins, losses, fencer_name) in enumerate([
            (all_left, left_wins, left_losses, '左剑手'),
            (all_right, right_wins, right_losses, '右剑手')
        ]):
            if not all_fencer:
                continue
                
            ax = axes[fencer_idx]
            
            characteristics = [
                ('维持安全距离', lambda f: f.get('maintained_safe_distance', False)),
                ('距离过近', lambda f: not f.get('maintained_safe_distance', False)),
                ('间距一致', lambda f: f.get('consistent_spacing', False)),
                ('间距不一致', lambda f: not f.get('consistent_spacing', False)),
                ('有反击机会', lambda f: f.get('counter_opportunities', 0) > 0),
                ('无反击机会', lambda f: f.get('counter_opportunities', 0) == 0),
                ('把握反击', lambda f: f.get('counter_taken', 0) > 0),
                ('错失反击', lambda f: f.get('counter_opportunities', 0) > 0 and f.get('counter_taken', 0) == 0),
                ('高防守比率', lambda f: f.get('defense_ratio', 0) > 0.7),
                ('低防守比率', lambda f: f.get('defense_ratio', 0) <= 0.7),
                ('距离接近最佳', lambda f: abs(f.get('defense_distance', 0) - 2.0) <= 0.3),
                ('距离偏离最佳', lambda f: abs(f.get('defense_distance', 0) - 2.0) > 0.3)
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
            
            ax.set_title(f'{fencer_name} - 防守动作效果分析')
            ax.axis('off')
        
        plt.tight_layout()
        correlation_path = os.path.join(output_dir, 'defense_action_outcome_correlation.png')
        plt.savefig(correlation_path, dpi=300, bbox_inches='tight')
        plt.close()
        chart_paths['defense_action_outcome_correlation'] = correlation_path
        
    except Exception as e:
        logging.error(f"Error creating defense action-outcome correlation: {e}")
    
    # Generate performance radar chart
    try:
        radar_chart_path = create_performance_radar_chart(defense_bouts, 'defense', output_dir)
        chart_paths['performance_radar'] = radar_chart_path
        logging.info(f"Generated defense performance radar chart: {radar_chart_path}")
    except Exception as e:
        logging.error(f"Error generating defense radar chart: {e}")
    
    return chart_paths

def process_defense_bouts_with_winners(analysis_dir: str, bout_classifications: List[Dict]) -> Dict[str, Any]:
    """
    Process all match analysis files and extract defense bout details with winner information.
    
    Args:
        analysis_dir: Directory containing match_analysis JSON files
        bout_classifications: List of bout classifications with winner info
    
    Returns:
        Dictionary with processed defense data including winner correlations
    """
    defense_bouts = []
    
    if not os.path.exists(analysis_dir):
        logging.warning(f"Analysis directory not found: {analysis_dir}")
        return {'bouts': [], 'summary': {}}
    
    # Create winner lookup and per-side defense flags
    winner_lookup = {}
    defense_lookup = {}
    for classification in bout_classifications:
        match_idx = classification.get('match_idx')
        if match_idx is not None:
            winner_lookup[match_idx] = classification.get('winner', 'undetermined')
            defense_lookup[match_idx] = {
                'left_is_defense': classification.get('left_category') == 'defense',
                'right_is_defense': classification.get('right_category') == 'defense'
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
            
            # Only process if this is a defense bout
            if any(classification.get('match_idx') == match_idx and 
                   (classification.get('left_category') == 'defense' or 
                    classification.get('right_category') == 'defense')
                   for classification in bout_classifications):
                
                bout_details = extract_defense_bout_details(data, winner)
                # Attach per-side flags so charts can filter correctly
                flags = defense_lookup.get(match_idx, {})
                bout_details['left_is_defense'] = bool(flags.get('left_is_defense'))
                bout_details['right_is_defense'] = bool(flags.get('right_is_defense'))
                if bout_details:
                    defense_bouts.append(bout_details)
            
        except Exception as e:
            logging.error(f"Error processing {filename}: {e}")
            continue
    
    # Generate summary statistics
    summary = generate_defense_summary_statistics(defense_bouts)
    
    return {
        'bouts': defense_bouts,
        'summary': summary,
        'total_defense_bouts': len(defense_bouts)
    }

def generate_defense_summary_statistics(defense_bouts: List[Dict]) -> Dict[str, Any]:
    """
    Generate comprehensive summary statistics for defense bouts including win/loss analysis.
    
    Args:
        defense_bouts: List of processed defense bout data with winner information
    
    Returns:
        Dictionary with summary statistics and comparisons
    """
    if not defense_bouts:
        return {}
    
    def analyze_side(side_key):
        wins = []
        losses = []
        all_data = []
        
        for bout in defense_bouts:
            fencer_data = bout.get(side_key, {})
            all_data.append(fencer_data)
            
            if fencer_data.get('is_winner') == True:
                wins.append(fencer_data)
            elif fencer_data.get('is_winner') == False:
                losses.append(fencer_data)
        
        total_decided = len(wins) + len(losses)
        win_rate = (len(wins) / total_decided) if total_decided > 0 else 0
        
        # Calculate averages
        def calc_avg(data_list, key, default=0.0):
            values = [d.get(key, default) for d in data_list if d.get(key) is not None]
            return np.mean(values) if values else default
        
        def calc_rate(data_list, key):
            count = sum(1 for d in data_list if d.get(key, False))
            return count / len(data_list) if data_list else 0
        
        return {
            'total_wins': len(wins),
            'total_losses': len(losses),
            'win_rate': win_rate,
            'avg_defense_distance': calc_avg(all_data, 'defense_distance'),
            'safe_distance_rate': calc_rate(all_data, 'maintained_safe_distance'),
            'consistent_spacing_rate': calc_rate(all_data, 'consistent_spacing'),
            'total_counter_opportunities': sum(d.get('counter_opportunities', 0) for d in all_data),
            'total_counter_taken': sum(d.get('counter_taken', 0) for d in all_data),
            'counter_success_rate': sum(d.get('counter_taken', 0) for d in all_data) / sum(d.get('counter_opportunities', 0) for d in all_data) if sum(d.get('counter_opportunities', 0) for d in all_data) > 0 else 0,
            'avg_defense_ratio': calc_avg(all_data, 'defense_ratio'),
            'win_avg_distance': calc_avg(wins, 'defense_distance'),
            'loss_avg_distance': calc_avg(losses, 'defense_distance'),
            'win_counter_rate': sum(d.get('counter_taken', 0) for d in wins) / sum(d.get('counter_opportunities', 0) for d in wins) if sum(d.get('counter_opportunities', 0) for d in wins) > 0 else 0,
            'loss_counter_rate': sum(d.get('counter_taken', 0) for d in losses) / sum(d.get('counter_opportunities', 0) for d in losses) if sum(d.get('counter_opportunities', 0) for d in losses) > 0 else 0
        }
    
    left_summary = analyze_side('left_fencer')
    right_summary = analyze_side('right_fencer')
    
    return {
        'total_bouts': len(defense_bouts),
        'left_fencer': left_summary,
        'right_fencer': right_summary
    }