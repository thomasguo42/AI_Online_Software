#!/usr/bin/env python3
"""
Test script to verify the adaptation between new bout_analysis data structure 
and fencer_analysis requirements.
"""

import os
import sys
import json
import logging
from pathlib import Path

# Add the your_scripts directory to the path
sys.path.append('your_scripts')

from fencer_analysis import load_bout_data, adapt_new_bout_data_structure

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def test_adaptation():
    """Test the adaptation function with sample data."""
    
    # Create sample data that mimics the new bout_analysis structure
    sample_data = {
        'match_idx': 1,
        'fps': 30,
        'frame_range': [0, 150],
        'left_data': {
            'movement_data': {
                'advance_intervals': [{'start': 10, 'end': 30}, {'start': 50, 'end': 80}],
                'pause_intervals': [{'start': 31, 'end': 49}],
                'retreat_intervals': [{'start': 81, 'end': 90}]
            },
            'summary_metrics': {
                'avg_velocity': 1.2,
                'avg_acceleration': 0.8,
                'has_launch': True,
                'launch_frame': 75
            },
            'launches': [
                {
                    'start_frame': 75,
                    'end_frame': 85,
                    'front_foot_max_velocity': 2.5,
                    'front_hip_max_velocity': 1.8
                }
            ],
            'extensions': [
                {
                    'start_frame': 60,
                    'end_frame': 85,
                    'duration_seconds': 0.83,
                    'max_velocity': 3.2
                }
            ],
            'has_launch': True,
            'launch_frame': 75,
            'velocity': 1.2,
            'acceleration': 0.8
        },
        'right_data': {
            'movement_data': {
                'advance_intervals': [{'start': 5, 'end': 25}],
                'pause_intervals': [{'start': 26, 'end': 45}],
                'retreat_intervals': [{'start': 85, 'end': 100}]
            },
            'summary_metrics': {
                'avg_velocity': 0.9,
                'avg_acceleration': 0.6,
                'has_launch': False,
                'launch_frame': -1
            },
            'launches': [],
            'extensions': [
                {
                    'start_frame': 40,
                    'end_frame': 55,
                    'duration_seconds': 0.5,
                    'max_velocity': 2.1
                }
            ],
            'has_launch': False,
            'launch_frame': -1,
            'velocity': 0.9,
            'acceleration': 0.6
        }
    }
    
    print("Original data structure:")
    print(json.dumps(sample_data, indent=2))
    
    # Test the adaptation
    adapted_data = adapt_new_bout_data_structure(sample_data)
    
    print("\n" + "="*50)
    print("Adapted data structure:")
    print(json.dumps(adapted_data, indent=2))
    
    # Verify required fields are present
    required_fields = [
        'advance', 'pause', 'arm_extensions', 'has_launch', 'launch_frame',
        'velocity', 'acceleration', 'attack_analysis', 'first_step'
    ]
    
    print("\n" + "="*50)
    print("Verification of required fields:")
    
    for side in ['left', 'right']:
        print(f"\n{side.upper()} FENCER:")
        side_data = adapted_data[f'{side}_data']
        
        for field in required_fields:
            if field in side_data:
                print(f"  ✓ {field}: {type(side_data[field])}")
                if field == 'attack_analysis':
                    attack_analysis = side_data[field]
                    print(f"    - all_launches: {len(attack_analysis.get('all_launches', []))} items")
                    print(f"    - all_extensions: {len(attack_analysis.get('all_extensions', []))} items")
                elif field in ['advance', 'pause', 'arm_extensions']:
                    print(f"    - {len(side_data[field])} intervals")
            else:
                print(f"  ✗ {field}: MISSING")
    
    return adapted_data

def test_with_real_data(analysis_dir, match_data_dir):
    """Test with real data if available."""
    if not os.path.exists(analysis_dir):
        print(f"Analysis directory {analysis_dir} not found. Skipping real data test.")
        return
    
    print(f"\n{'='*50}")
    print("Testing with real data...")
    
    try:
        bout_data = load_bout_data(analysis_dir, match_data_dir)
        print(f"Successfully loaded {len(bout_data)} bouts")
        
        if bout_data:
            # Check first bout
            first_bout = bout_data[0]
            print(f"\nFirst bout (match_{first_bout['match_idx']}):")
            
            for side in ['left', 'right']:
                side_data = first_bout[f'{side}_data']
                print(f"\n{side.upper()} FENCER:")
                print(f"  - advance intervals: {len(side_data.get('advance', []))}")
                print(f"  - pause intervals: {len(side_data.get('pause', []))}")
                print(f"  - arm extensions: {len(side_data.get('arm_extensions', []))}")
                print(f"  - has_launch: {side_data.get('has_launch', False)}")
                print(f"  - velocity: {side_data.get('velocity', 0)}")
                print(f"  - attack_analysis: {'present' if 'attack_analysis' in side_data else 'missing'}")
                
                if 'attack_analysis' in side_data:
                    aa = side_data['attack_analysis']
                    print(f"    - launches: {len(aa.get('all_launches', []))}")
                    print(f"    - extensions: {len(aa.get('all_extensions', []))}")
        
    except Exception as e:
        print(f"Error testing with real data: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    print("Testing bout_analysis to fencer_analysis adaptation...")
    
    # Test with sample data
    adapted_data = test_adaptation()
    
    # Test with real data if paths are provided
    if len(sys.argv) >= 3:
        analysis_dir = sys.argv[1]
        match_data_dir = sys.argv[2]
        test_with_real_data(analysis_dir, match_data_dir)
    else:
        print("\nTo test with real data, run:")
        print("python test_adaptation.py <analysis_dir> <match_data_dir>")
    
    print("\nTest completed!") 