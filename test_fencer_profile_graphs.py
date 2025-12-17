#!/usr/bin/env python3
"""
Test script to demonstrate fencer profile graph generation.
"""

import sys
import os
import json

# Add the your_scripts directory to the path
sys.path.append('/workspace/Project/your_scripts')

from fencer_profile_plotting import save_fencer_profile_plots

def create_sample_fencer_data():
    """Create sample fencer data for testing."""
    return {
        "fencer_id": "Fencer_Left",
        "metrics": {
            "avg_first_step_init": 0.155,
            "avg_first_step_velocity": 1.159,
            "avg_first_step_acceleration": 3.813,
            "avg_velocity": 0.642,
            "std_velocity": 0.952,
            "avg_acceleration": 6.084,
            "std_acceleration": 9.614,
            "avg_advance_ratio": 0.654,
            "avg_pause_ratio": 0.346,
            "total_arm_extensions": 9,
            "avg_arm_extension_duration": 0.467,
            "avg_launch_promptness": 0.433,
            "attacking_ratio": 0.25,
            "avg_right_of_way_score": 1.05
        },
        "bouts": [
            {
                "match_idx": 1,
                "metrics": {
                    "velocity": 2.377,
                    "acceleration": 28.757,
                    "advance_ratio": 1.0,
                    "pause_ratio": 0.0,
                    "attacking_score": 70.0,
                    "interval_analysis": {
                        "summary": {
                            "attacks": {"total": 2, "simple": 1, "compound": 0, "holding": 1, "preparations": 0},
                            "tempo": {"steady": 2, "variable": 1, "broken": 0},
                            "distance": {"good_attack_distances": 1, "missed_opportunities": 1},
                            "defense": {"good_distance_management": 2, "counter_opportunities": 1, "counters_executed": 0}
                        }
                    }
                },
                "is_winner": True
            },
            {
                "match_idx": 2,
                "metrics": {
                    "velocity": 1.611,
                    "acceleration": 9.422,
                    "advance_ratio": 0.156,
                    "pause_ratio": 0.844,
                    "attacking_score": 36.24,
                    "interval_analysis": {
                        "summary": {
                            "attacks": {"total": 1, "simple": 1, "compound": 0, "holding": 0, "preparations": 0},
                            "tempo": {"steady": 3, "variable": 0, "broken": 0},
                            "distance": {"good_attack_distances": 0, "missed_opportunities": 0},
                            "defense": {"good_distance_management": 1, "counter_opportunities": 1, "counters_executed": 0}
                        }
                    }
                },
                "is_winner": False
            },
            {
                "match_idx": 3,
                "metrics": {
                    "velocity": 2.15,
                    "acceleration": 15.3,
                    "advance_ratio": 0.65,
                    "pause_ratio": 0.35,
                    "attacking_score": 55.8,
                    "interval_analysis": {
                        "summary": {
                            "attacks": {"total": 3, "simple": 2, "compound": 1, "holding": 0, "preparations": 0},
                            "tempo": {"steady": 1, "variable": 2, "broken": 0},
                            "distance": {"good_attack_distances": 2, "missed_opportunities": 1},
                            "defense": {"good_distance_management": 1, "counter_opportunities": 2, "counters_executed": 1}
                        }
                    }
                },
                "is_winner": True
            }
        ]
    }

def test_fencer_profile_graphs():
    """Test the fencer profile graph generation."""
    print("Testing Fencer Profile Graph Generation")
    print("=" * 50)
    
    # Create sample data
    left_fencer_data = create_sample_fencer_data()
    
    # Create right fencer data with different characteristics
    right_fencer_data = create_sample_fencer_data()
    right_fencer_data["fencer_id"] = "Fencer_Right"
    right_fencer_data["metrics"]["avg_first_step_init"] = 0.067  # Faster reaction
    right_fencer_data["metrics"]["avg_velocity"] = 3.585  # Higher velocity
    right_fencer_data["metrics"]["avg_acceleration"] = 28.484  # Higher acceleration
    right_fencer_data["metrics"]["avg_advance_ratio"] = 0.353  # Lower forward pressure
    right_fencer_data["metrics"]["attacking_ratio"] = 0.75  # Higher attack ratio
    
    # Modify right fencer bout data to show different performance
    for bout in right_fencer_data["bouts"]:
        bout["metrics"]["velocity"] *= 2.5
        bout["metrics"]["acceleration"] *= 1.8
        bout["is_winner"] = not bout.get("is_winner", False)  # Flip win/loss pattern
    
    # Test output directory
    test_output_dir = "/workspace/Project/test_fencer_plots"
    os.makedirs(test_output_dir, exist_ok=True)
    
    # Generate plots for both fencers
    fencer_data_map = {
        "Fencer_Left": left_fencer_data,
        "Fencer_Right": right_fencer_data
    }
    
    results = {}
    
    for fencer_name, fencer_data in fencer_data_map.items():
        print(f"\nGenerating plots for {fencer_name}...")
        fencer_output_dir = os.path.join(test_output_dir, fencer_name)
        
        try:
            plot_files = save_fencer_profile_plots(fencer_data, fencer_name, fencer_output_dir)
            results[fencer_name] = plot_files
            
            print(f"✅ Successfully generated {len(plot_files)} plot files for {fencer_name}")
            for plot_type, file_path in plot_files.items():
                print(f"   📊 {plot_type}: {file_path}")
                
        except Exception as e:
            print(f"❌ Error generating plots for {fencer_name}: {str(e)}")
    
    # Print summary
    print(f"\n" + "=" * 50)
    print("GRAPH GENERATION SUMMARY")
    print("=" * 50)
    
    print(f"\nGenerated graphs for {len(results)} fencers:")
    for fencer_name, plot_files in results.items():
        print(f"\n{fencer_name}:")
        print(f"  📈 Comprehensive Profile: Available")
        print(f"  🎯 Radar Chart: Available") 
        print(f"  📊 Total Plot Files: {len(plot_files)}")
    
    print(f"\nOutput Directory: {test_output_dir}")
    print("\nGraph Types Generated:")
    print("  1. 🎯 Radar Chart - Overall performance profile")
    print("  2. ⚔️  Attack Type Analysis - Attack frequency and success")
    print("  3. 🎵 Tempo Analysis - Tempo distribution")
    print("  4. 📊 Performance Metrics - Key performance indicators")
    print("  5. 📏 Distance Management - Attack and defense distance effectiveness")
    print("  6. 📈 Bout Progression - Performance trends over time")
    print("  7. 🏷️  Tag Analysis - Strengths and weaknesses")
    print("  8. 🏆 Victory Analysis - Win/loss patterns")
    
    return results

if __name__ == "__main__":
    results = test_fencer_profile_graphs()
    print(f"\n🎉 Test completed! Generated graphs for {len(results)} fencers.")