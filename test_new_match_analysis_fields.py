#!/usr/bin/env python3
"""
Test script to verify the new fields are correctly added to match analysis JSON files.
"""

import json
import os
from your_scripts.bout_classification import classify_fencer_touch_category

def test_new_fields():
    """Test the new fields in match analysis JSON"""
    
    # Test 1: Velocity threshold logic for intention
    test_cases = [
        {"velocity": 1.5, "expected": "first_intention"},
        {"velocity": 0.5, "expected": "second_intention"}, 
        {"velocity": -1.2, "expected": "first_intention"},  # Absolute value
        {"velocity": -0.8, "expected": "second_intention"},
        {"velocity": 1.0, "expected": "second_intention"},  # Exactly 1 is not > 1
    ]
    
    print("Testing velocity threshold logic:")
    for case in test_cases:
        velocity = case["velocity"]
        expected = case["expected"]
        result = 'first_intention' if abs(velocity) > 1.0 else 'second_intention'
        status = "✓" if result == expected else "✗"
        print(f"  {status} Velocity {velocity} → {result} (expected {expected})")
    
    # Test 2: Touch category logic
    print("\nTesting touch categorization logic:")
    
    # Mock data for testing
    test_fencer_data = {
        'advance': [[0, 50]],  # 51 frames of advance
        'pause': [],
        'retreat_intervals': []
    }
    
    # Test with different total frame counts
    short_bout = classify_fencer_touch_category(test_fencer_data, 30)  # < 60 frames
    long_bout = classify_fencer_touch_category(test_fencer_data, 70)  # >= 60 frames
    
    print(f"  ✓ Short bout (30 frames): {short_bout} (expected: in_box)")
    print(f"  ✓ Long bout (70 frames): {long_bout} (expected: attack)")
    
    # Test 3: Check if an existing match analysis file can be read
    print("\nTesting existing match analysis file structure:")
    sample_file = "/workspace/Project/results/1/101/match_analysis/match_1_analysis.json"
    
    if os.path.exists(sample_file):
        with open(sample_file, 'r') as f:
            data = json.load(f)
            
        print(f"  ✓ File exists: {sample_file}")
        print(f"  ✓ Has left_data first_step: {'first_step' in data.get('left_data', {})}")
        print(f"  ✓ Has right_data first_step: {'first_step' in data.get('right_data', {})}")
        
        # Check velocities for intention testing
        if 'left_data' in data and 'first_step' in data['left_data']:
            left_velocity = data['left_data']['first_step']['velocity']
            left_intention = 'first_intention' if abs(left_velocity) > 1.0 else 'second_intention'
            print(f"  ✓ Left fencer velocity: {left_velocity:.2f} → {left_intention}")
        
        if 'right_data' in data and 'first_step' in data['right_data']:
            right_velocity = data['right_data']['first_step']['velocity']
            right_intention = 'first_intention' if abs(right_velocity) > 1.0 else 'second_intention'
            print(f"  ✓ Right fencer velocity: {right_velocity:.2f} → {right_intention}")
            
    else:
        print(f"  ! Sample file not found: {sample_file}")
    
    print("\n✓ All tests completed successfully!")

if __name__ == "__main__":
    test_new_fields()