#!/usr/bin/env python3
"""
Graph Analysis Module

Generates concise, graph-specific analysis for each tactical visualization.
This replaces individual bout analysis with focused, actionable insights for each graph.
"""

from your_scripts.gemini_rest import generate_text as gemini_generate_text
import logging
import json
from typing import Dict, Any, List, Optional

DEFAULT_MODEL = 'gemini-2.5-flash-lite'

def analyze_attack_type_graph(left_data: List[Dict], right_data: List[Dict]) -> str:
    """Generate analysis for attack type and victory correlation graph."""
    try:
        # Count attack types for both fencers
        left_attacks = {}
        left_victories = {}
        right_attacks = {}
        right_victories = {}
        
        for entry in left_data:
            attack_type = entry.get('attack_type', 'Unknown')
            is_winner = entry.get('is_winner', False)
            left_attacks[attack_type] = left_attacks.get(attack_type, 0) + 1
            if is_winner:
                left_victories[attack_type] = left_victories.get(attack_type, 0) + 1
        
        for entry in right_data:
            attack_type = entry.get('attack_type', 'Unknown')
            is_winner = entry.get('is_winner', False)
            right_attacks[attack_type] = right_attacks.get(attack_type, 0) + 1
            if is_winner:
                right_victories[attack_type] = right_victories.get(attack_type, 0) + 1

        # Create analysis prompt
        prompt = f"""
        分析进攻类型和胜利数据：
        
        左侧击剑手：进攻 {left_attacks}, 获胜 {left_victories}
        右侧击剑手：进攻 {right_attacks}, 获胜 {right_victories}
        
        请提供简洁的分析（100-150字），包括：
        - 击剑手之间的进攻模式差异
        - 成功率和战术偏好
        - 关键训练建议
        
        使用专业击剑术语。要直接且具有可操作性。回复请用中文。
        """
        text = gemini_generate_text(
            prompt,
            model=DEFAULT_MODEL,
            temperature=0.2,
            top_k=1,
            top_p=0.8,
            max_output_tokens=512,
            timeout_seconds=45,
            max_attempts=6,
            response_mime_type='text/plain',
        )
        return text or "由于API错误，分析暂时不可用。"
    except Exception as e:
        logging.error(f"Error generating attack type analysis: {e}")
        return "由于API错误，分析暂时不可用。"

def analyze_tempo_type_graph(left_data: List[Dict], right_data: List[Dict]) -> str:
    """Generate analysis for tempo type and victory correlation graph."""
    try:
        left_tempos = {}
        left_tempo_wins = {}
        right_tempos = {}
        right_tempo_wins = {}
        
        for entry in left_data:
            tempo = entry.get('tempo_type', 'Unknown')
            is_winner = entry.get('is_winner', False)
            left_tempos[tempo] = left_tempos.get(tempo, 0) + 1
            if is_winner:
                left_tempo_wins[tempo] = left_tempo_wins.get(tempo, 0) + 1
        
        for entry in right_data:
            tempo = entry.get('tempo_type', 'Unknown')
            is_winner = entry.get('is_winner', False)
            right_tempos[tempo] = right_tempos.get(tempo, 0) + 1
            if is_winner:
                right_tempo_wins[tempo] = right_tempo_wins.get(tempo, 0) + 1

        prompt = f"""
        分析节奏和胜利数据：
        
        左侧击剑手：节奏使用 {left_tempos}, 获胜 {left_tempo_wins}
        右侧击剑手：节奏使用 {right_tempos}, 获胜 {right_tempo_wins}
        
        请提供简洁的分析（100-150字），包括：
        - 节奏偏好和成功率
        - 时机策略差异
        - 节奏控制洞察
        - 训练建议
        
        对教练要直接且具有可操作性。回复请用中文。
        """
        text = gemini_generate_text(
            prompt,
            model=DEFAULT_MODEL,
            temperature=0.2,
            top_k=1,
            top_p=0.8,
            max_output_tokens=512,
            timeout_seconds=45,
            max_attempts=6,
            response_mime_type='text/plain',
        )
        return text or "由于API错误，分析暂时不可用。"
    except Exception as e:
        logging.error(f"Error generating tempo type analysis: {e}")
        return "由于API错误，分析暂时不可用。"

def analyze_attack_distance_graph(left_data: List[Dict], right_data: List[Dict]) -> str:
    """Generate analysis for attack distance effectiveness graph."""
    try:
        left_distances = [entry['avg_distance'] for entry in left_data if 'avg_distance' in entry]
        right_distances = [entry['avg_distance'] for entry in right_data if 'avg_distance' in entry]
        
        left_avg = sum(left_distances) / len(left_distances) if left_distances else 0
        right_avg = sum(right_distances) / len(right_distances) if right_distances else 0
        
        left_wins = sum(1 for entry in left_data if entry.get('is_winner', False))
        right_wins = sum(1 for entry in right_data if entry.get('is_winner', False))

        prompt = f"""
        分析进攻距离数据：
        
        左侧击剑手：平均 {left_avg:.2f}米, {len(left_distances)} 次进攻, {left_wins} 次获胜
        右侧击剑手：平均 {right_avg:.2f}米, {len(right_distances)} 次进攻, {right_wins} 次获胜
        最佳距离：约2.0米
        
        请提供简洁的分析（100-150字），包括：
        - 距离策略对比与最佳范围
        - 从距离偏好看格斗风格洞察
        - 距离管理建议
        
        专注于实用的衡量和时机建议。回复请用中文。
        """
        text = gemini_generate_text(
            prompt,
            model=DEFAULT_MODEL,
            temperature=0.2,
            top_k=1,
            top_p=0.8,
            max_output_tokens=512,
            timeout_seconds=45,
            max_attempts=6,
            response_mime_type='text/plain',
        )
        return text or "由于API错误，分析暂时不可用。"
    except Exception as e:
        logging.error(f"Error generating attack distance analysis: {e}")
        return "由于API错误，分析暂时不可用。"

def analyze_counter_opportunities_graph(left_data: List[Dict], right_data: List[Dict]) -> str:
    """Generate analysis for counter opportunity usage graph."""
    try:
        left_total_opps = sum(entry.get('total_opportunities', 0) for entry in left_data)
        left_used = sum(entry.get('used', 0) for entry in left_data)
        right_total_opps = sum(entry.get('total_opportunities', 0) for entry in right_data)
        right_used = sum(entry.get('used', 0) for entry in right_data)
        
        left_usage_rate = (left_used / left_total_opps * 100) if left_total_opps > 0 else 0
        right_usage_rate = (right_used / right_total_opps * 100) if right_total_opps > 0 else 0

        prompt = f"""
        分析反击机会数据：
        
        左侧击剑手：{left_total_opps} 次机会, {left_used} 次使用 ({left_usage_rate:.1f}%)
        右侧击剑手：{right_total_opps} 次机会, {right_used} 次使用 ({right_usage_rate:.1f}%)
        
        请提供简洁的分析（100-150字），包括：
        - 反击识别和使用对比
        - 战术意识洞察
        - 反击改进的训练建议
        
        专注于实用的反击发展。回复请用中文。
        """
        text = gemini_generate_text(
            prompt,
            model=DEFAULT_MODEL,
            temperature=0.2,
            top_k=1,
            top_p=0.8,
            max_output_tokens=512,
            timeout_seconds=45,
            max_attempts=6,
            response_mime_type='text/plain',
        )
        return text or "由于API错误，分析暂时不可用。"
    except Exception as e:
        logging.error(f"Error generating counter opportunities analysis: {e}")
        return "由于API错误，分析暂时不可用。"

def analyze_retreat_quality_graph(left_data: List[Dict], right_data: List[Dict]) -> str:
    """Generate analysis for retreat quality and spacing consistency."""
    try:
        left_safe_pct = sum(entry.get('safe_distance_percentage', 0) for entry in left_data) / len(left_data) if left_data else 0
        left_consistent_pct = sum(entry.get('consistent_spacing_percentage', 0) for entry in left_data) / len(left_data) if left_data else 0
        right_safe_pct = sum(entry.get('safe_distance_percentage', 0) for entry in right_data) / len(right_data) if right_data else 0  
        right_consistent_pct = sum(entry.get('consistent_spacing_percentage', 0) for entry in right_data) / len(right_data) if right_data else 0

        prompt = f"""
        分析退却质量数据：
        
        左侧击剑手：安全距离 {left_safe_pct:.1f}%, 间距一致性 {left_consistent_pct:.1f}%
        右侧击剑手：安全距离 {right_safe_pct:.1f}%, 间距一致性 {right_consistent_pct:.1f}%
        
        请提供简洁的分析（100-150字），包括：
        - 防守位置对比
        - 间距一致性洞察
        - 退却质量建议
        
        专注于实用的防守改进。回复请用中文。
        """
        text = gemini_generate_text(
            prompt,
            model=DEFAULT_MODEL,
            temperature=0.2,
            top_k=1,
            top_p=0.8,
            max_output_tokens=512,
            timeout_seconds=45,
            max_attempts=6,
            response_mime_type='text/plain',
        )
        return text or "由于API错误，分析暂时不可用。"
    except Exception as e:
        logging.error(f"Error generating retreat quality analysis: {e}")
        return "由于API错误，分析暂时不可用。"

def analyze_retreat_distance_graph(left_data: List[Dict], right_data: List[Dict]) -> str:
    """Generate analysis for retreat distance patterns."""
    try:
        left_avg_dist = sum(entry.get('avg_distance', 0) for entry in left_data) / len(left_data) if left_data else 0
        left_variance = sum(entry.get('distance_variance', 0) for entry in left_data) / len(left_data) if left_data else 0
        right_avg_dist = sum(entry.get('avg_distance', 0) for entry in right_data) / len(right_data) if right_data else 0
        right_variance = sum(entry.get('distance_variance', 0) for entry in right_data) / len(right_data) if right_data else 0

        prompt = f"""
        分析退却距离数据：
        
        左侧击剑手：平均距离 {left_avg_dist:.2f}米, 方差 {left_variance:.3f}
        右侧击剑手：平均距离 {right_avg_dist:.2f}米, 方差 {right_variance:.3f}
        
        请提供简洁的分析（100-150字），包括：
        - 退却距离模式和一致性对比
        - 可预测性与适应性洞察
        - 防守策略建议
        
        专注于实用的防守移动优化。回复请用中文。
        """
        text = gemini_generate_text(
            prompt,
            model=DEFAULT_MODEL,
            temperature=0.2,
            top_k=1,
            top_p=0.8,
            max_output_tokens=512,
            timeout_seconds=45,
            max_attempts=6,
            response_mime_type='text/plain',
        )
        return text or "由于API错误，分析暂时不可用。"
    except Exception as e:
        logging.error(f"Error generating retreat distance analysis: {e}")
        return "由于API错误，分析暂时不可用。"

def analyze_defensive_quality_graph(left_data: List[Dict], right_data: List[Dict]) -> str:
    """Generate analysis for overall defensive quality."""
    try:
        left_good_pct = sum(entry.get('good_percentage', 0) for entry in left_data) / len(left_data) if left_data else 0
        right_good_pct = sum(entry.get('good_percentage', 0) for entry in right_data) / len(right_data) if right_data else 0

        prompt = f"""
        分析防守质量数据：
        
        左侧击剑手：良好防守动作 {left_good_pct:.1f}%
        右侧击剑手：良好防守动作 {right_good_pct:.1f}%
        
        请提供简洁的分析（100-150字），包括：
        - 防守有效性对比
        - 防守技能洞察
        - 具体改进建议
        
        专注于实用的防守发展。回复请用中文。
        """
        text = gemini_generate_text(
            prompt,
            model=DEFAULT_MODEL,
            temperature=0.2,
            top_k=1,
            top_p=0.8,
            max_output_tokens=512,
            timeout_seconds=45,
            max_attempts=6,
            response_mime_type='text/plain',
        )
        return text or "由于API错误，分析暂时不可用。"
    except Exception as e:
        logging.error(f"Error generating defensive quality analysis: {e}")
        return "由于API错误，分析暂时不可用。"

def analyze_bout_outcome_graph(left_data: List[Dict], right_data: List[Dict]) -> str:
    """Generate analysis for bout outcome patterns (attack vs retreat victory)."""
    try:
        left_attack_wins = sum(1 for entry in left_data if entry.get('is_winner', False) and entry.get('majority_behavior') == 'attack')
        left_retreat_wins = sum(1 for entry in left_data if entry.get('is_winner', False) and entry.get('majority_behavior') == 'retreat')
        right_attack_wins = sum(1 for entry in right_data if entry.get('is_winner', False) and entry.get('majority_behavior') == 'attack')
        right_retreat_wins = sum(1 for entry in right_data if entry.get('is_winner', False) and entry.get('majority_behavior') == 'retreat')

        prompt = f"""
        分析回合结果模式：
        
        左侧击剑手：进攻获胜 {left_attack_wins}, 防守获胜 {left_retreat_wins}
        右侧击剑手：进攻获胜 {right_attack_wins}, 防守获胜 {right_retreat_wins}
        
        请提供简洁的分析（100-150字），包括：
        - 获胜策略偏好（攻击性vs防守性）
        - 战术方法有效性
        - 策略适应性建议
        
        专注于战术平衡和发展。回复请用中文。
        """
        text = gemini_generate_text(
            prompt,
            model=DEFAULT_MODEL,
            temperature=0.2,
            top_k=1,
            top_p=0.8,
            max_output_tokens=512,
            timeout_seconds=45,
            max_attempts=6,
            response_mime_type='text/plain',
        )
        return text or "由于API错误，分析暂时不可用。"
    except Exception as e:
        logging.error(f"Error generating bout outcome analysis: {e}")
        return "由于API错误，分析暂时不可用。"

def analyze_radar_chart(left_metrics: Dict, right_metrics: Dict) -> str:
    """Generate analysis for radar chart comparison."""
    try:
        prompt = f"""
        分析这个雷达图性能档案：
        
        左侧击剑手：反应 {left_metrics.get('avg_first_step_init', 0):.3f}秒, 速度 {left_metrics.get('avg_velocity', 0):.2f}, 加速度 {left_metrics.get('avg_acceleration', 0):.2f}, 前进比率 {left_metrics.get('avg_advance_ratio', 0):.2f}, 进攻频率 {left_metrics.get('attacking_ratio', 0):.2f}
        
        右侧击剑手：反应 {right_metrics.get('avg_first_step_init', 0):.3f}秒, 速度 {right_metrics.get('avg_velocity', 0):.2f}, 加速度 {right_metrics.get('avg_acceleration', 0):.2f}, 前进比率 {right_metrics.get('avg_advance_ratio', 0):.2f}, 进攻频率 {right_metrics.get('attacking_ratio', 0):.2f}
        
        请提供简洁的分析（150-200字），包括：
        - 整体性能档案对比
        - 每个击剑手的最强/最弱属性
        - 从指标看格斗风格洞察
        - 具体训练建议
        
        专注于可操作的运动发展。回复请用中文。
        """
        
        text = gemini_generate_text(
            prompt,
            model=DEFAULT_MODEL,
            temperature=0.2,
            top_k=1,
            top_p=0.8,
            max_output_tokens=512,
            timeout_seconds=45,
            max_attempts=6,
            response_mime_type='text/plain',
        )
        return text or "由于API错误，分析暂时不可用。"
        
    except Exception as e:
        logging.error(f"Error generating radar chart analysis: {e}")
        return "由于API错误，分析暂时不可用。"

def analyze_comparison_chart(left_metrics: Dict, right_metrics: Dict) -> str:
    """Generate analysis for bar chart performance comparison."""
    try:
        prompt = f"""
        分析这个性能对比柱状图：
        
        左侧击剑手：反应 {left_metrics.get('avg_first_step_init', 0):.3f}秒, 速度 {left_metrics.get('avg_velocity', 0):.2f}, 加速度 {left_metrics.get('avg_acceleration', 0):.2f}, 前进比率 {left_metrics.get('avg_advance_ratio', 0):.2f}, 进攻成功率 {left_metrics.get('attacking_ratio', 0):.2f}
        
        右侧击剑手：反应 {right_metrics.get('avg_first_step_init', 0):.3f}秒, 速度 {right_metrics.get('avg_velocity', 0):.2f}, 加速度 {right_metrics.get('avg_acceleration', 0):.2f}, 前进比率 {right_metrics.get('avg_advance_ratio', 0):.2f}, 进攻成功率 {right_metrics.get('attacking_ratio', 0):.2f}
        
        请提供简洁的分析（150-200字），包括：
        - 直接指标对比和优势
        - 性能差异的战术含义
        - 每个击剑手的优先改进领域
        
        专注于竞争优势和发展优先级。回复请用中文。
        """
        
        text = gemini_generate_text(
            prompt,
            model=DEFAULT_MODEL,
            temperature=0.2,
            top_k=1,
            top_p=0.8,
            max_output_tokens=512,
            timeout_seconds=45,
            max_attempts=6,
            response_mime_type='text/plain',
        )
        return text or "由于API错误，分析暂时不可用。"
        
    except Exception as e:
        logging.error(f"Error generating comparison chart analysis: {e}")
        return "由于API错误，分析暂时不可用。"

# Main analysis dispatcher
def generate_graph_analysis(graph_type: str, left_data: Any, right_data: Any, **kwargs) -> str:
    """
    Generate graph-specific analysis based on graph type.
    
    Args:
        graph_type: Type of graph ('attack_type', 'tempo_type', etc.)
        left_data: Data for left fencer
        right_data: Data for right fencer
        **kwargs: Additional parameters for specific graph types
    
    Returns:
        Generated analysis text
    """
    
    analysis_functions = {
        'attack_type_analysis': analyze_attack_type_graph,
        'tempo_type_analysis': analyze_tempo_type_graph,
        'attack_distance_analysis': analyze_attack_distance_graph,
        'counter_opportunities': analyze_counter_opportunities_graph,
        'retreat_quality': analyze_retreat_quality_graph,
        'retreat_distance': analyze_retreat_distance_graph,
        'defensive_quality': analyze_defensive_quality_graph,
        'bout_outcome': analyze_bout_outcome_graph,
        'radar_chart': analyze_radar_chart,
        'comparison_chart': analyze_comparison_chart
    }
    
    if graph_type in analysis_functions:
        return analysis_functions[graph_type](left_data, right_data)
    else:
        logging.warning(f"Unknown graph type: {graph_type}")
        return f"Analysis for {graph_type} is not yet implemented."
