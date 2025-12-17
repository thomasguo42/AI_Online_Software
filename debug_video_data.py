#!/usr/bin/env python3

import sys
import os
import json
sys.path.append('/workspace/Project')

from your_scripts.video_view_analysis import get_basic_video_data

def test_video_data():
    # Test with sample upload
    upload_id = 1
    user_id = 1
    
    print("Testing get_basic_video_data...")
    
    try:
        result = get_basic_video_data(upload_id, user_id)
        
        print(f"Success: {result['success']}")
        
        if result['success']:
            print("\nKeys in result:", list(result.keys()))
            
            if 'detailed_analysis' in result:
                print("\nDetailed analysis keys:", list(result['detailed_analysis'].keys()))
                
                # Check in_box data structure
                if 'in_box' in result['detailed_analysis']:
                    in_box = result['detailed_analysis']['in_box']
                    print(f"\nIn-box structure:")
                    print(f"  Keys: {list(in_box.keys())}")
                    
                    if 'left_fencer' in in_box:
                        print(f"  Left fencer data: {in_box['left_fencer']}")
                    
                    if 'display_data' in in_box:
                        print(f"  Display data keys: {list(in_box['display_data'].keys())}")
                        if 'left_fencer' in in_box['display_data']:
                            print(f"  Left display data: {in_box['display_data']['left_fencer']}")
            else:
                print("No detailed_analysis in result!")
                
            # Check what other keys might be missing
            missing_keys = []
            expected_keys = ['radar_data', 'bout_type_stats', 'total_touches', 'detailed_analysis']
            for key in expected_keys:
                if key not in result:
                    missing_keys.append(key)
            
            if missing_keys:
                print(f"Missing expected keys: {missing_keys}")
        else:
            print(f"Error: {result.get('error', 'Unknown error')}")
            
    except Exception as e:
        print(f"Exception: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_video_data()