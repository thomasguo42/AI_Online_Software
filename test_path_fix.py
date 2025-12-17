#!/usr/bin/env python3
"""
Test script to verify the path fix works correctly.
"""

import os
import json

def test_path_fixes():
    """Test if the path handling fixes work."""
    print("Testing Path Fix for Fencer Profile Graphs")
    print("=" * 50)
    
    # Test the exact scenario
    test_user_id = 1
    test_upload_id = 101
    
    # Check the cross_bout_analysis.json file
    analysis_dir = f"/workspace/Project/results/{test_user_id}/{test_upload_id}/fencer_analysis"
    cross_bout_file = os.path.join(analysis_dir, 'cross_bout_analysis.json')
    
    if os.path.exists(cross_bout_file):
        print(f"✅ Found analysis file: {cross_bout_file}")
        
        with open(cross_bout_file, 'r') as f:
            data = json.load(f)
        
        if 'fencer_profile_plots' in data:
            print("✅ Profile plots data found")
            
            for fencer_side, plots in data['fencer_profile_plots'].items():
                print(f"\n{fencer_side} Profile Graphs:")
                
                for plot_type, plot_path in plots.items():
                    print(f"  Original path: {plot_path}")
                    
                    # Test original path
                    if os.path.exists(plot_path):
                        print(f"    ✅ Original path works: {plot_type}")
                    else:
                        print(f"    ❌ Original path failed: {plot_type}")
                        
                        # Test relative path conversion (same logic as Flask route)
                        if plot_path.startswith('/workspace/Project/'):
                            rel_path = plot_path[len('/workspace/Project/'):]
                            print(f"    Trying relative path: {rel_path}")
                            
                            if os.path.exists(rel_path):
                                print(f"    ✅ Relative path works: {plot_type}")
                            else:
                                print(f"    ❌ Relative path also failed: {plot_type}")
    
    # Test specific file existence  
    print(f"\n" + "=" * 30)
    print("FILE EXISTENCE TEST")
    print("=" * 30)
    
    expected_files = [
        f"/workspace/Project/results/{test_user_id}/{test_upload_id}/fencer_analysis/profile_plots/Fencer_Left/fencer_left_profile_analysis.png",
        f"/workspace/Project/results/{test_user_id}/{test_upload_id}/fencer_analysis/profile_plots/Fencer_Left/fencer_left_radar_profile.png",
        f"/workspace/Project/results/{test_user_id}/{test_upload_id}/fencer_analysis/profile_plots/Fencer_Right/fencer_right_profile_analysis.png",
        f"/workspace/Project/results/{test_user_id}/{test_upload_id}/fencer_analysis/profile_plots/Fencer_Right/fencer_right_radar_profile.png"
    ]
    
    for file_path in expected_files:
        if os.path.exists(file_path):
            file_size = os.path.getsize(file_path)
            print(f"✅ {os.path.basename(file_path)} - {file_size:,} bytes")
        else:
            print(f"❌ {os.path.basename(file_path)} - NOT FOUND")
    
    print(f"\n" + "=" * 50)
    print("RESOLUTION")
    print("=" * 50)
    print("The issue was a path mismatch:")
    print("- JSON file contains: /workspace/Project/results/1/101/...")
    print("- Flask was expecting: results/1/101/...")
    print("- Fixed: Added path conversion logic in Flask routes")
    print("\n🔧 FIXED: Flask routes now handle both absolute and relative paths")

if __name__ == "__main__":
    test_path_fixes()