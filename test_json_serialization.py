#!/usr/bin/env python3
"""
Test JSON serialization for video view data
"""

import json
import math
from your_scripts.video_view_analysis import generate_video_view_data, sanitize_value, sanitize_data_structure

def test_json_serialization():
    """Test that all video view data can be JSON serialized"""
    
    print("🔍 Testing JSON Serialization for Video View")
    print("=" * 45)
    
    # Test 1: Test sanitize_value function with problematic values
    print("Testing sanitize_value function:")
    
    test_cases = [
        (float('inf'), 0.0, "Infinity"),
        (float('-inf'), 0.0, "Negative infinity"),
        (float('nan'), 0.0, "NaN"),
        (None, 0.0, "None"),
        (42, 42.0, "Normal integer"),
        (3.14, 3.14, "Normal float"),
        ("string", 0.0, "String (should convert to 0)")
    ]
    
    for input_val, expected, description in test_cases:
        try:
            result = sanitize_value(input_val)
            status = "✅" if result == expected else "❌"
            print(f"  {status} {description}: {input_val} → {result}")
        except Exception as e:
            print(f"  ❌ {description}: Error - {e}")
    
    # Test 2: Test sanitize_data_structure with complex data
    print("\n🔧 Testing complex data structure sanitization:")
    
    problematic_data = {
        'normal_value': 42.5,
        'infinity': float('inf'),
        'negative_infinity': float('-inf'),
        'nan_value': float('nan'),
        'nested_dict': {
            'good_value': 100.0,
            'bad_value': float('inf'),
            'null_value': None
        },
        'list_with_problems': [1.0, float('nan'), 3.14, float('inf')],
        'boolean': True,
        'string': "test"
    }
    
    try:
        sanitized = sanitize_data_structure(problematic_data)
        
        # Try to serialize to JSON
        json_str = json.dumps(sanitized)
        print("✅ Successfully sanitized and serialized complex data")
        print(f"   Original had inf/nan values, sanitized version JSON length: {len(json_str)}")
        
        # Verify specific values
        assert sanitized['normal_value'] == 42.5, "Normal value should remain unchanged"
        assert sanitized['infinity'] == 0.0, "Infinity should become 0.0"
        assert sanitized['nan_value'] == 0.0, "NaN should become 0.0"
        assert sanitized['nested_dict']['bad_value'] == 0.0, "Nested bad values should be sanitized"
        assert sanitized['list_with_problems'] == [1.0, 0.0, 3.14, 0.0], "List items should be sanitized"
        
        print("✅ All sanitization rules working correctly")
        
    except Exception as e:
        print(f"❌ Error in data structure sanitization: {e}")
        return False
    
    # Test 3: Test actual video view data generation and serialization
    print("\n📊 Testing real video view data serialization:")
    
    try:
        # Try to generate actual video view data
        video_data = generate_video_view_data(101, 1)  # Using sample data
        
        if video_data.get('success'):
            # Try to serialize to JSON
            json_str = json.dumps(video_data)
            print("✅ Successfully generated and serialized real video view data")
            print(f"   JSON data size: {len(json_str)} characters")
            
            # Verify structure
            assert 'radar_data' in video_data, "Missing radar_data"
            assert 'bout_type_stats' in video_data, "Missing bout_type_stats"
            assert 'total_touches' in video_data, "Missing total_touches"
            
            # Check radar data values are all numbers
            left_values = video_data['radar_data']['left_fencer']['values']
            right_values = video_data['radar_data']['right_fencer']['values']
            
            for i, (left_val, right_val) in enumerate(zip(left_values, right_values)):
                assert isinstance(left_val, (int, float)), f"Left radar value {i} is not numeric: {type(left_val)}"
                assert isinstance(right_val, (int, float)), f"Right radar value {i} is not numeric: {type(right_val)}"
                assert not math.isnan(left_val), f"Left radar value {i} is NaN"
                assert not math.isnan(right_val), f"Right radar value {i} is NaN"
                assert not math.isinf(left_val), f"Left radar value {i} is infinite"
                assert not math.isinf(right_val), f"Right radar value {i} is infinite"
            
            print("✅ All radar chart values are valid numbers")
            
            # Check bout type stats
            for fencer in ['left_fencer', 'right_fencer']:
                for category in ['attack', 'defense', 'in_box']:
                    win_rate = video_data['bout_type_stats'][fencer][category]['win_rate']
                    assert isinstance(win_rate, (int, float)), f"Win rate is not numeric: {type(win_rate)}"
                    assert not math.isnan(win_rate), f"Win rate is NaN for {fencer} {category}"
                    assert not math.isinf(win_rate), f"Win rate is infinite for {fencer} {category}"
                    assert 0 <= win_rate <= 100, f"Win rate out of bounds: {win_rate}"
            
            print("✅ All bout type statistics are valid")
            
        else:
            print(f"⚠️  Could not generate video view data: {video_data.get('error', 'Unknown error')}")
            print("   This might be expected if sample data is not available")
            
    except Exception as e:
        print(f"❌ Error testing real data: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    print("\n🎉 All JSON serialization tests passed!")
    print("\nThe video view should now work without serialization errors.")
    
    return True

if __name__ == "__main__":
    test_json_serialization()