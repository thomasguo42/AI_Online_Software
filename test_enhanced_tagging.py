#!/usr/bin/env python3
"""
Test script to demonstrate the enhanced tagging system with negative/disadvantage tags.
"""

import json
import sys
import os

# Add the your_scripts directory to the path
sys.path.append('/workspace/Project/your_scripts')

from tagging import extract_tags_from_bout_analysis, get_predefined_tags

def test_tagging_with_sample_data():
    """Test the enhanced tagging system using the sample JSON data provided."""
    
    # Sample data from the user (simplified for testing)
    sample_data = {
        "left_data": {
            "advance": [[5, 19], [33, 38], [130, 142]],
            "pause": [[0, 4], [20, 32], [39, 129], [143, 182], [183, 217]],
            "arm_extensions": [[137.0, 142.0]],
            "has_launch": True,
            "launch_frame": 140.0,
            "velocity": 1.6111535589595676,
            "acceleration": 9.422250614199122,
            "arm_extension_freq": 1,
            "advance_ratio": 0.1559633027522936,
            "pause_ratio": 0.8440366972477065,
            "first_step": {
                "init_time": 0.16666666666666666,
                "velocity": 1.8766614011634442,
                "acceleration": 4.2076638649573885
            },
            "interval_analysis": {
                "advance_analyses": [
                    {
                        "attack_info": {
                            "has_attack": False,
                            "attack_type": "no_attack"
                        },
                        "tempo_type": "steady_tempo",
                        "tempo_changes": 3,
                        "good_attack_distance": False
                    },
                    {
                        "attack_info": {
                            "has_attack": True,
                            "attack_type": "simple_attack"
                        },
                        "tempo_type": "steady_tempo",
                        "tempo_changes": 0,
                        "good_attack_distance": False
                    }
                ],
                "retreat_analyses": [
                    {
                        "defensive_quality": "good",
                        "maintained_safe_distance": False,
                        "consistent_spacing": True,
                        "opportunities_missed": 1,
                        "successful_distance_pulls": 0,
                        "launch_responses": [{"launch_frame": 206}]
                    }
                ],
                "summary": {
                    "attacks": {"total": 1, "simple": 1, "compound": 0, "holding": 0, "preparations": 0},
                    "tempo": {"steady": 3, "variable": 0, "broken": 0}
                }
            }
        },
        "right_data": {
            "advance": [[0, 24], [166, 217]],
            "pause": [[25, 32], [33, 43], [44, 50], [51, 84], [85, 110], [111, 141], [142, 165]],
            "arm_extensions": [[20.0, 24.0], [166.0, 214.0]],
            "has_launch": True,
            "launch_frame": 206.0,
            "velocity": 3.585455012657034,
            "acceleration": 28.483689137938363,
            "arm_extension_freq": 2,
            "advance_ratio": 0.3532110091743119,
            "pause_ratio": 0.6467889908256881,
            "first_step": {
                "init_time": 0.06666666666666667,
                "velocity": 1.829930662045367,
                "acceleration": 7.841101658820599
            },
            "interval_analysis": {
                "advance_analyses": [
                    {
                        "attack_info": {
                            "has_attack": True,
                            "attack_type": "simple_preparation"
                        },
                        "tempo_type": "steady_tempo",
                        "tempo_changes": 8,
                        "good_attack_distance": False
                    },
                    {
                        "attack_info": {
                            "has_attack": True,
                            "attack_type": "holding_attack"
                        },
                        "tempo_type": "steady_tempo", 
                        "tempo_changes": 13,
                        "good_attack_distance": True
                    }
                ],
                "retreat_analyses": [
                    {
                        "defensive_quality": "good",
                        "maintained_safe_distance": True,
                        "consistent_spacing": True,
                        "opportunities_missed": 0,
                        "successful_distance_pulls": 0
                    }
                ],
                "summary": {
                    "attacks": {"total": 2, "simple": 0, "compound": 0, "holding": 1, "preparations": 1},
                    "tempo": {"steady": 2, "variable": 0, "broken": 0}
                }
            }
        }
    }
    
    # Extract tags using the enhanced system
    print("Testing Enhanced Fencing Tagging System")
    print("=" * 50)
    
    tags = extract_tags_from_bout_analysis(sample_data)
    
    print("\nLeft Fencer Tags:")
    print("-" * 20)
    for tag in sorted(tags['left']):
        print(f"  • {tag}")
    
    print(f"\nTotal Left Fencer Tags: {len(tags['left'])}")
    
    print("\nRight Fencer Tags:")
    print("-" * 20)
    for tag in sorted(tags['right']):
        print(f"  • {tag}")
    
    print(f"\nTotal Right Fencer Tags: {len(tags['right'])}")
    
    print("\nAll Available Tags in System:")
    print("-" * 30)
    all_tags = get_predefined_tags()
    
    # Organize tags by category
    positive_tags = [tag for tag in all_tags if not any(neg in tag for neg in ['no_', 'poor_', 'excessive_', 'slow_', 'low_', 'missed_', 'failed_', 'limited_', 'insufficient_', 'inconsistent_'])]
    negative_tags = [tag for tag in all_tags if tag not in positive_tags]
    
    print(f"\nPositive Performance Tags ({len(positive_tags)}):")
    for tag in sorted(positive_tags):
        print(f"  ✓ {tag}")
    
    print(f"\nNegative/Disadvantage Tags ({len(negative_tags)}):")
    for tag in sorted(negative_tags):
        print(f"  ✗ {tag}")
    
    print(f"\nTotal Available Tags: {len(all_tags)}")
    
    # Show comparison between left and right fencer
    print("\n" + "=" * 50)
    print("FENCER COMPARISON")
    print("=" * 50)
    
    left_negative_tags = [tag for tag in tags['left'] if tag in negative_tags]
    right_negative_tags = [tag for tag in tags['right'] if tag in negative_tags]
    
    print(f"\nLeft Fencer Disadvantages ({len(left_negative_tags)}):")
    for tag in sorted(left_negative_tags):
        print(f"  ⚠️  {tag}")
        
    print(f"\nRight Fencer Disadvantages ({len(right_negative_tags)}):")
    for tag in sorted(right_negative_tags):
        print(f"  ⚠️  {tag}")
    
    # Performance summary
    print(f"\nPerformance Summary:")
    print(f"Left Fencer:  {len(tags['left']) - len(left_negative_tags)} strengths, {len(left_negative_tags)} weaknesses")
    print(f"Right Fencer: {len(tags['right']) - len(right_negative_tags)} strengths, {len(right_negative_tags)} weaknesses")

if __name__ == "__main__":
    test_tagging_with_sample_data()