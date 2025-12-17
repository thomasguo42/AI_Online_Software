#!/usr/bin/env python3
"""
Test script to verify the new fencer-centric architecture is working correctly.
"""

import os
import sys
sys.path.append('/workspace/Project')

def test_new_architecture():
    """Test the new fencer-centric architecture"""
    print("Testing New Fencer-Centric Architecture")
    print("=" * 60)
    
    # Test 1: Check if profiles were generated
    fencer_profiles_dir = "/workspace/Project/fencer_profiles/1"
    if os.path.exists(fencer_profiles_dir):
        print("✅ Fencer profiles directory exists")
        
        # Check each fencer
        for fencer_dir in os.listdir(fencer_profiles_dir):
            fencer_path = os.path.join(fencer_profiles_dir, fencer_dir)
            if os.path.isdir(fencer_path):
                print(f"\n📁 Fencer {fencer_dir}:")
                
                # Check profile data
                profile_data = os.path.join(fencer_path, 'profile_data.json')
                if os.path.exists(profile_data):
                    print(f"    ✅ Profile data exists")
                else:
                    print(f"    ❌ Missing profile data")
                
                # Check plots directory
                plots_dir = os.path.join(fencer_path, 'profile_plots')
                if os.path.exists(plots_dir):
                    plot_files = os.listdir(plots_dir)
                    print(f"    ✅ Plots directory exists ({len(plot_files)} files)")
                    for plot in plot_files:
                        file_size = os.path.getsize(os.path.join(plots_dir, plot))
                        print(f"      - {plot}: {file_size:,} bytes")
                else:
                    print(f"    ❌ Missing plots directory")
    else:
        print("❌ Fencer profiles directory does not exist")
    
    # Test 2: Check Flask route mapping
    print(f"\n" + "=" * 30)
    print("FLASK ROUTE MAPPING")
    print("=" * 30)
    
    graph_types = ['radar_profile', 'profile_analysis']
    test_fencer_id = 1
    
    for graph_type in graph_types:
        expected_path = f"fencer_profiles/1/{test_fencer_id}/profile_plots/fencer_{test_fencer_id}_{graph_type}.png"
        if os.path.exists(expected_path):
            print(f"✅ {graph_type}: Route will serve {expected_path}")
        else:
            print(f"❌ {graph_type}: File not found at {expected_path}")
    
    # Test 3: Architecture Summary
    print(f"\n" + "=" * 60)
    print("ARCHITECTURE SUMMARY")
    print("=" * 60)
    print("✅ OLD (Upload-centric): results/1/101/fencer_analysis/profile_plots/")
    print("✅ NEW (Fencer-centric): fencer_profiles/1/{fencer_id}/profile_plots/")
    print("")
    print("🔧 BENEFITS OF NEW ARCHITECTURE:")
    print("   • Graphs are independent of specific uploads/matches")
    print("   • Aggregated data across ALL fencer performances")
    print("   • Cleaner directory structure")
    print("   • Easier to manage and maintain")
    print("   • Better fits the conceptual model (fencer profiles, not match profiles)")
    
    print(f"\n🚀 READY FOR TESTING:")
    print(f"   1. Start Flask: python app.py")
    print(f"   2. Login as user '1234'")
    print(f"   3. Go to Fencer Management")
    print(f"   4. Click 'View Profile' for any fencer")
    print(f"   5. Profile graphs should now load from fencer-centric directories!")

if __name__ == "__main__":
    test_new_architecture()