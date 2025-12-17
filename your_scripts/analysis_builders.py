#!/usr/bin/env python3
"""
Analysis builders for creating structured DataFrames from bout analysis data.
These DataFrames are optimized for visualization and statistical analysis.
"""

import pandas as pd
import numpy as np
import logging

def extract_fencer_analysis_data(bout_data):
    """
    Extract analysis data for the 8 specific graphs requested.
    Returns separate data for left and right fencers.
    """
    left_data = {
        'attack_types': [],           # Graph 1: attack type and victory
        'tempo_types': [],            # Graph 2: tempo type and victory  
        'attack_distances': [],       # Graph 3: average attack distance analysis
        'counter_opportunities': [],  # Graph 4: counter opportunities (moved up)
        'retreat_quality': [],        # Graph 5: safe distance and spacing consistency
        'retreat_distances': [],      # Graph 6: average retreat distance analysis
        'defensive_quality': [],      # Graph 7: defensive quality
        'bout_outcomes': []           # Graph 8: bout wins through attack/retreat
    }
    
    right_data = {
        'attack_types': [],
        'tempo_types': [],
        'attack_distances': [],
        'counter_opportunities': [],
        'retreat_quality': [],
        'retreat_distances': [],
        'defensive_quality': [],
        'bout_outcomes': []
    }
    
    for bout_idx, bout in enumerate(bout_data):
        winner_side = bout.get('winner_side', 'unknown')
        
        # Process each fencer's data
        for side, fencer_data in [('left', left_data), ('right', right_data)]:
            side_data = bout.get(f'{side}_data', {})
            interval_analysis = side_data.get('interval_analysis', {})
            movement_data = side_data.get('movement_data', {})
            
            is_winner = (winner_side == side)
            
            # Graph 1: Attack Type Analysis
            advance_analyses = interval_analysis.get('advance_analyses', [])
            for advance in advance_analyses:
                attack_info = advance.get('attack_info', {})
                if attack_info.get('has_attack', False):
                    attack_type = attack_info.get('attack_type', 'Unknown')
                    fencer_data['attack_types'].append({
                        'bout_idx': bout_idx,
                        'attack_type': attack_type,
                        'is_winner': is_winner
                    })
            
            # Graph 2: Tempo Type Analysis
            for advance in advance_analyses:
                tempo_type = advance.get('tempo_type', 'Unknown')
                fencer_data['tempo_types'].append({
                    'bout_idx': bout_idx,
                    'tempo_type': tempo_type,
                    'is_winner': is_winner
                })
            
            # Graph 3: Average Attack Distance Analysis
            attack_distances = []
            for advance in advance_analyses:
                attack_info = advance.get('attack_info', {})
                if attack_info.get('has_attack', False):
                    avg_distance = advance.get('avg_distance', 0)
                    min_distance = advance.get('min_distance', 0)
                    # Use average distance, fallback to min_distance if avg not available
                    distance = avg_distance if avg_distance > 0 else min_distance
                    if distance > 0:
                        attack_distances.append(distance)
            
            if attack_distances:
                # Calculate overall average for this bout
                bout_avg_distance = sum(attack_distances) / len(attack_distances)
                fencer_data['attack_distances'].append({
                    'bout_idx': bout_idx,
                    'avg_distance': bout_avg_distance,
                    'num_attacks': len(attack_distances),
                    'is_winner': is_winner
                })
            
            # Graph 4: Counter Opportunities Analysis (moved from Graph 5)
            retreat_analyses = interval_analysis.get('retreat_analyses', [])
            total_counter_opportunities = 0
            counter_used = 0
            counter_missed = 0
            
            for retreat in retreat_analyses:
                counter_opps = retreat.get('counter_opportunities', [])
                for opp in counter_opps:
                    total_counter_opportunities += 1
                    if opp.get('action_taken', False):
                        counter_used += 1
                    else:
                        counter_missed += 1
                
                # Also check the summary fields
                opportunities_taken = retreat.get('opportunities_taken', 0)
                opportunities_missed = retreat.get('opportunities_missed', 0)
                counter_used += opportunities_taken
                counter_missed += opportunities_missed
                total_counter_opportunities += opportunities_taken + opportunities_missed
            
            fencer_data['counter_opportunities'].append({
                'bout_idx': bout_idx,
                'total_opportunities': total_counter_opportunities,
                'used': counter_used,
                'missed': counter_missed,
                'is_winner': is_winner
            })
            
            # Graph 5: Retreat Quality Analysis (Spacing Consistency Only)
            total_retreats = len(retreat_analyses)
            safe_distance_retreats = 0
            consistent_spacing_retreats = 0
            
            for retreat in retreat_analyses:
                if retreat.get('maintained_safe_distance', False):
                    safe_distance_retreats += 1
                if retreat.get('consistent_spacing', False):
                    consistent_spacing_retreats += 1
            
            if total_retreats > 0:
                fencer_data['retreat_quality'].append({
                    'bout_idx': bout_idx,
                    'total_retreats': total_retreats,
                    'safe_distance': safe_distance_retreats,
                    'consistent_spacing': consistent_spacing_retreats,
                    'safe_distance_percentage': (safe_distance_retreats / total_retreats) * 100,
                    'consistent_spacing_percentage': (consistent_spacing_retreats / total_retreats) * 100,
                    'is_winner': is_winner
                })
            
            # Graph 6: Average Retreat Distance Analysis
            retreat_distances = []
            retreat_variances = []
            
            for retreat in retreat_analyses:
                avg_distance = retreat.get('avg_distance', 0)
                distance_variance = retreat.get('distance_variance', 0)
                if avg_distance > 0:
                    retreat_distances.append(avg_distance)
                if distance_variance >= 0:
                    retreat_variances.append(distance_variance)
            
            if retreat_distances:
                # Calculate overall average for this bout
                bout_avg_retreat_distance = sum(retreat_distances) / len(retreat_distances)
                bout_avg_variance = sum(retreat_variances) / len(retreat_variances) if retreat_variances else 0
                fencer_data['retreat_distances'].append({
                    'bout_idx': bout_idx,
                    'avg_distance': bout_avg_retreat_distance,
                    'distance_variance': bout_avg_variance,
                    'num_retreats': len(retreat_distances),
                    'is_winner': is_winner
                })
            
            # Graph 7: Defensive Quality Analysis
            total_defensive_actions = len(retreat_analyses)
            good_defensive_actions = 0
            poor_defensive_actions = 0
            
            for retreat in retreat_analyses:
                defensive_quality = retreat.get('defensive_quality', 'unknown')
                if defensive_quality.lower() in ['good', 'excellent', 'strong']:
                    good_defensive_actions += 1
                elif defensive_quality.lower() in ['poor', 'bad', 'weak']:
                    poor_defensive_actions += 1
            
            if total_defensive_actions > 0:
                fencer_data['defensive_quality'].append({
                    'bout_idx': bout_idx,
                    'total_actions': total_defensive_actions,
                    'good_actions': good_defensive_actions,
                    'poor_actions': poor_defensive_actions,
                    'good_percentage': (good_defensive_actions / total_defensive_actions) * 100,
                    'is_winner': is_winner
                })
            
            # Graph 8: Bout Outcome Analysis (Attack vs Retreat Victory)
            # Determine if fencer won through attack or retreat based on majority frames
            advance_intervals = movement_data.get('advance_intervals', [])
            retreat_intervals = movement_data.get('retreat_intervals', [])
            pause_intervals = movement_data.get('pause_intervals', [])
            
            # Calculate total frames in each type
            advance_frames = sum([(interval[1] - interval[0] + 1) for interval in advance_intervals])
            retreat_frames = sum([(interval[1] - interval[0] + 1) for interval in retreat_intervals])
            pause_frames = sum([(interval[1] - interval[0] + 1) for interval in pause_intervals])
            
            # Determine majority behavior
            total_frames = advance_frames + retreat_frames + pause_frames
            if total_frames > 0:
                advance_ratio = advance_frames / total_frames
                retreat_ratio = (retreat_frames + pause_frames) / total_frames
                
                majority_behavior = 'attack' if advance_ratio > retreat_ratio else 'retreat'
            else:
                majority_behavior = 'unknown'
            
            fencer_data['bout_outcomes'].append({
                'bout_idx': bout_idx,
                'is_winner': is_winner,
                'majority_behavior': majority_behavior,
                'advance_frames': advance_frames,
                'retreat_frames': retreat_frames + pause_frames,
                'advance_ratio': advance_ratio if total_frames > 0 else 0,
                'retreat_ratio': retreat_ratio if total_frames > 0 else 0
            })
    
    logging.info(f"Extracted analysis data for {len(bout_data)} bouts")
    logging.info(f"Left fencer - Attack types: {len(left_data['attack_types'])}, "
                f"Tempo types: {len(left_data['tempo_types'])}, "
                f"Attack distances: {len(left_data['attack_distances'])}, "
                f"Counter opportunities: {len(left_data['counter_opportunities'])}, "
                f"Retreat quality: {len(left_data['retreat_quality'])}, "
                f"Retreat distances: {len(left_data['retreat_distances'])}, "
                f"Defensive quality: {len(left_data['defensive_quality'])}, "
                f"Bout outcomes: {len(left_data['bout_outcomes'])}")
    
    return left_data, right_data 