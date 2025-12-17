#!/usr/bin/env python3
"""
Debug script to check database state for fencer profile integration.
"""

import sys
sys.path.append('/workspace/Project')

from app import create_app
from models import db, User, Fencer, Upload
import os

def debug_database():
    """Debug the database state to understand the issue."""
    print("Debugging Database State for Fencer Profile Graphs")
    print("=" * 60)
    
    # Create Flask app context
    app = create_app()
    
    with app.app_context():
        # Check users
        print("USERS:")
        users = User.query.all()
        for user in users:
            print(f"  User {user.id}: {user.username}")
        
        print(f"\nFENCERS:")
        fencers = Fencer.query.all()
        for fencer in fencers:
            print(f"  Fencer {fencer.id}: {fencer.name} (User {fencer.user_id})")
        
        print(f"\nUPLOADS:")
        uploads = Upload.query.all()
        for upload in uploads:
            print(f"  Upload {upload.id}: User {upload.user_id}, Left: {upload.left_fencer_id}, Right: {upload.right_fencer_id}, Status: {upload.status}")
            
            # Check if analysis exists
            analysis_dir = f"/workspace/Project/results/{upload.user_id}/{upload.id}/fencer_analysis"
            if os.path.exists(analysis_dir):
                print(f"    ✅ Analysis directory exists: {analysis_dir}")
                
                cross_bout_file = os.path.join(analysis_dir, 'cross_bout_analysis.json')
                if os.path.exists(cross_bout_file):
                    print(f"    ✅ Cross-bout analysis exists")
                    
                    # Check for profile plots
                    profile_plot_dir = os.path.join(analysis_dir, 'profile_plots')
                    if os.path.exists(profile_plot_dir):
                        print(f"    ✅ Profile plots directory exists")
                        for fencer_dir in ['Fencer_Left', 'Fencer_Right']:
                            fencer_plot_dir = os.path.join(profile_plot_dir, fencer_dir)
                            if os.path.exists(fencer_plot_dir):
                                files = os.listdir(fencer_plot_dir)
                                print(f"      {fencer_dir}: {files}")
                    else:
                        print(f"    ❌ No profile plots directory")
                else:
                    print(f"    ❌ No cross-bout analysis")
            else:
                print(f"    ❌ No analysis directory: {analysis_dir}")
        
        print(f"\n" + "=" * 60)
        print("ISSUE ANALYSIS")
        print("=" * 60)
        
        # Check the specific case we're testing
        test_upload_id = 101
        upload = Upload.query.filter_by(id=test_upload_id).first()
        
        if upload:
            print(f"Upload 101 found: User {upload.user_id}")
            print(f"Left fencer: {upload.left_fencer_id}, Right fencer: {upload.right_fencer_id}")
            
            # Check if fencers exist
            left_fencer = Fencer.query.get(upload.left_fencer_id) if upload.left_fencer_id else None
            right_fencer = Fencer.query.get(upload.right_fencer_id) if upload.right_fencer_id else None
            
            print(f"Left fencer object: {left_fencer.name if left_fencer else 'None'}")
            print(f"Right fencer object: {right_fencer.name if right_fencer else 'None'}")
            
            # Test the Flask route logic for finding graphs
            if left_fencer:
                print(f"\nTesting graph detection for Left Fencer ({left_fencer.name}):")
                uploads_for_fencer = Upload.query.filter(
                    (Upload.left_fencer_id == left_fencer.id) | (Upload.right_fencer_id == left_fencer.id)
                ).filter_by(user_id=upload.user_id, status='completed').all()
                
                print(f"  Found {len(uploads_for_fencer)} uploads for this fencer")
                
                for up in uploads_for_fencer:
                    analysis_dir = os.path.join('results', str(up.user_id), str(up.id), 'fencer_analysis')
                    print(f"  Checking: {analysis_dir}")
                    if os.path.exists(analysis_dir):
                        print(f"    ✅ Analysis dir exists")
                    else:
                        print(f"    ❌ Analysis dir missing")
        else:
            print(f"❌ Upload 101 not found in database!")

if __name__ == "__main__":
    debug_database()