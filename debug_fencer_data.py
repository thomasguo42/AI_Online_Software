#!/usr/bin/env python3
"""
Debug script to see what data exists for fencers and why profiles aren't generating.
"""

import os
import sys
import json
sys.path.append('/workspace/Project')

from app import create_app
from models import db, User, Fencer, Upload

def debug_fencer_data():
    """Debug what data exists for fencers"""
    print("Debugging Fencer Data and Profile Generation")
    print("=" * 60)
    
    app = create_app()
    
    with app.app_context():
        print("1. CHECKING ALL FENCERS AND THEIR UPLOADS")
        print("-" * 50)
        
        fencers = Fencer.query.all()
        for fencer in fencers:
            print(f"\n📁 Fencer: {fencer.name} (ID: {fencer.id}, User: {fencer.user_id})")
            
            # Find uploads for this fencer
            uploads = Upload.query.filter(
                (Upload.left_fencer_id == fencer.id) | (Upload.right_fencer_id == fencer.id)
            ).filter_by(user_id=fencer.user_id, status='completed').all()
            
            print(f"   Associated uploads: {len(uploads)}")
            
            for upload in uploads:
                side = 'left' if upload.left_fencer_id == fencer.id else 'right'
                print(f"   Upload {upload.id}: {fencer.name} is {side} fencer")
                
                # Check if analysis data exists
                result_dir = f"/workspace/Project/results/{upload.user_id}/{upload.id}"
                fencer_analysis_dir = os.path.join(result_dir, 'fencer_analysis')
                
                if os.path.exists(fencer_analysis_dir):
                    print(f"     ✅ Analysis dir exists: {fencer_analysis_dir}")
                    
                    # Check for fencer-specific analysis file
                    fencer_file = os.path.join(fencer_analysis_dir, f'fencer_Fencer_{side.title()}_analysis.json')
                    if os.path.exists(fencer_file):
                        try:
                            with open(fencer_file, 'r') as f:
                                data = json.load(f)
                            bout_count = len(data.get('bouts', []))
                            print(f"     ✅ Fencer analysis file: {bout_count} bouts")
                        except Exception as e:
                            print(f"     ❌ Error reading fencer analysis: {e}")
                    else:
                        print(f"     ❌ Missing fencer analysis file: {fencer_file}")
                    
                    # Check for cross-bout analysis
                    cross_bout_file = os.path.join(fencer_analysis_dir, 'cross_bout_analysis.json')
                    if os.path.exists(cross_bout_file):
                        print(f"     ✅ Cross-bout analysis exists")
                    else:
                        print(f"     ❌ Missing cross-bout analysis")
                        
                else:
                    print(f"     ❌ No analysis dir: {fencer_analysis_dir}")
        
        print(f"\n" + "=" * 60)
        print("2. TESTING PROFILE GENERATION FOR EACH FENCER")
        print("=" * 60)
        
        sys.path.insert(0, '/workspace/Project/your_scripts')
        from fencer_centric_profiles import collect_fencer_data_across_uploads
        
        for fencer in fencers[:2]:  # Test first 2 fencers
            print(f"\n🧪 Testing fencer: {fencer.name} (ID: {fencer.id})")
            
            try:
                fencer_data = collect_fencer_data_across_uploads(fencer.id, fencer.user_id)
                print(f"   Total bouts collected: {fencer_data.get('total_bouts', 0)}")
                print(f"   Total uploads: {fencer_data.get('total_uploads', 0)}")
                print(f"   Upload sources: {len(fencer_data.get('upload_sources', []))}")
                
                if fencer_data.get('all_bouts'):
                    print(f"   ✅ Has bout data - should generate graphs")
                else:
                    print(f"   ❌ No bout data - will create empty profile")
                    
            except Exception as e:
                print(f"   ❌ Error collecting data: {str(e)}")
        
        print(f"\n" + "=" * 60)
        print("3. CHECKING CURRENT PROFILE DIRECTORIES")
        print("=" * 60)
        
        profile_base = "/workspace/Project/fencer_profiles"
        if os.path.exists(profile_base):
            for user_dir in os.listdir(profile_base):
                user_path = os.path.join(profile_base, user_dir)
                if os.path.isdir(user_path):
                    print(f"\nUser {user_dir}:")
                    for fencer_dir in os.listdir(user_path):
                        fencer_path = os.path.join(user_path, fencer_dir)
                        if os.path.isdir(fencer_path):
                            plots_dir = os.path.join(fencer_path, 'profile_plots')
                            if os.path.exists(plots_dir):
                                files = os.listdir(plots_dir)
                                print(f"  Fencer {fencer_dir}: {len(files)} files")
                            else:
                                print(f"  Fencer {fencer_dir}: No plots directory")
        else:
            print("❌ No fencer_profiles directory exists")

if __name__ == "__main__":
    debug_fencer_data()