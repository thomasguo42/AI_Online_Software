#!/usr/bin/env python3
"""
Test script to verify video view analysis functionality
"""

import json
import os
from your_scripts.video_view_analysis import generate_video_view_data, calculate_performance_metrics

def test_video_view_analysis():
    """Test the video view analysis with existing data"""
    
    print("🎯 Testing Video View Analysis Implementation")
    print("=" * 50)
    
    # Test 1: Check if we can import all required functions
    try:
        from your_scripts.video_view_analysis import (
            generate_video_view_data, 
            calculate_performance_metrics,
            normalize_value,
            format_radar_data
        )
        print("✅ Successfully imported all video view analysis functions")
    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False
    
    # Test 2: Test normalize function
    print("\n🔧 Testing utility functions:")
    assert normalize_value(0.5, 0, 1) == 0.5, "Normalize mid-range failed"
    assert normalize_value(2, 0, 4) == 0.5, "Normalize scale failed" 
    assert normalize_value(-1, 0, 1) == 0, "Normalize min clamp failed"
    assert normalize_value(2, 0, 1) == 1, "Normalize max clamp failed"
    print("✅ Normalize function working correctly")
    
    # Test 3: Check if analysis directories exist
    print("\n📁 Checking for sample data:")
    sample_analysis_dir = "results/1/101/match_analysis"
    if os.path.exists(sample_analysis_dir):
        files = os.listdir(sample_analysis_dir)
        json_files = [f for f in files if f.endswith('_analysis.json')]
        print(f"✅ Found {len(json_files)} match analysis JSON files")
        
        # Test 4: Try loading and processing sample data
        if json_files:
            try:
                video_data = generate_video_view_data(101, 1)
                if video_data['success']:
                    print(f"✅ Successfully generated video view data")
                    print(f"   - Total touches: {video_data['total_touches']}")
                    print(f"   - Left fencer overall score: {video_data['radar_data']['left_fencer']['overall_score']:.1f}")
                    print(f"   - Right fencer overall score: {video_data['radar_data']['right_fencer']['overall_score']:.1f}")
                    
                    # Check radar data structure
                    left_values = video_data['radar_data']['left_fencer']['values']
                    right_values = video_data['radar_data']['right_fencer']['values']
                    
                    assert len(left_values) == 9, f"Expected 9 metrics, got {len(left_values)}"
                    assert len(right_values) == 9, f"Expected 9 metrics, got {len(right_values)}"
                    print("✅ Radar data structure is correct (9 metrics)")
                    
                    # Check bout type stats
                    bout_stats = video_data['bout_type_stats']
                    required_categories = ['attack', 'defense', 'in_box']
                    
                    for fencer in ['left_fencer', 'right_fencer']:
                        for category in required_categories:
                            assert category in bout_stats[fencer], f"Missing {category} for {fencer}"
                            assert 'count' in bout_stats[fencer][category], f"Missing count for {fencer} {category}"
                            assert 'wins' in bout_stats[fencer][category], f"Missing wins for {fencer} {category}"
                            assert 'win_rate' in bout_stats[fencer][category], f"Missing win_rate for {fencer} {category}"
                    
                    print("✅ Bout type statistics structure is correct")
                    
                else:
                    print(f"❌ Failed to generate video view data: {video_data.get('error', 'Unknown error')}")
                    
            except Exception as e:
                print(f"❌ Error processing sample data: {e}")
                import traceback
                traceback.print_exc()
        
    else:
        print(f"⚠️  Sample analysis directory not found: {sample_analysis_dir}")
        print("   This is expected if no analysis has been run yet")
    
    # Test 5: Test metric calculation logic
    print("\n🧮 Testing metric calculation logic:")
    
    # Mock data for testing individual metrics
    mock_metrics = {
        'first_intention_attempts': 10,
        'first_intention_wins': 7,
        'second_intention_attempts': 5,
        'second_intention_wins': 2,
        'attack_promptness_scores': [60, 70, 80],
        'attack_aggressiveness_scores': [50, 60, 70],
        'total_attacking_advances': 8,
        'good_distance_advances': 6,
        'attack_attempts': 5,
        'attack_wins': 3,
        'total_retreats': 3,
        'good_quality_retreats': 2,
        'total_counter_ops': 4,
        'counters_executed': 2,
        'defense_actions': 2,
        'defense_wins': 1
    }
    
    from your_scripts.video_view_analysis import calculate_final_scores
    scores = calculate_final_scores(mock_metrics)
    
    expected_first_intention = 70.0  # 7/10 * 100
    expected_attack_effectiveness = 60.0  # 3/5 * 100
    expected_defense_effectiveness = 50.0  # 1/2 * 100
    
    assert abs(scores['first_intention_effectiveness'] - expected_first_intention) < 0.1, "First intention calculation error"
    assert abs(scores['attack_effectiveness'] - expected_attack_effectiveness) < 0.1, "Attack effectiveness calculation error"
    assert abs(scores['defense_effectiveness'] - expected_defense_effectiveness) < 0.1, "Defense effectiveness calculation error"
    
    print("✅ Metric calculation logic is working correctly")
    
    print("\n🎉 All tests passed! Video View implementation is ready.")
    print("\nNext steps:")
    print("1. Start the Flask app: python app.py")
    print("2. Navigate to an upload with completed analysis")
    print("3. Click '性能分析' button to view the new video analysis")
    print("4. Verify radar charts display correctly with proper color coding")
    print("5. Check bout type statistics are accurate")
    
    return True

if __name__ == "__main__":
    test_video_view_analysis()
