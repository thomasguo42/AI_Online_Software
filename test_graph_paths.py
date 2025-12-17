#!/usr/bin/env python3
"""
Test script to verify graph path generation and file serving
"""

import os
import sys
sys.path.append('/workspace/Project')

from app import create_app
from models import Fencer, Upload

def test_graph_paths():
    app = create_app()
    
    with app.app_context():
        print("=== Testing Graph Path Generation ===")
        
        # Test path construction
        result_folder = app.config['RESULT_FOLDER']
        print(f"RESULT_FOLDER: {result_folder}")
        
        # Find a fencer with existing graphs
        fencers = Fencer.query.all()
        print(f"Found {len(fencers)} fencers")
        
        for fencer in fencers[:3]:  # Test first 3 fencers
            print(f"\n--- Testing Fencer {fencer.id}: {fencer.name} ---")
            
            # Check for graph files
            analysis_dir = os.path.join(result_folder, str(fencer.user_id), 'fencer', str(fencer.id))
            radar_file = os.path.join(analysis_dir, f'fencer_{fencer.id}_radar_profile.png')
            analysis_file = os.path.join(analysis_dir, f'fencer_{fencer.id}_profile_analysis.png')
            
            print(f"Analysis dir: {analysis_dir}")
            print(f"Radar file exists: {os.path.exists(radar_file)}")
            print(f"Analysis file exists: {os.path.exists(analysis_file)}")
            
            if os.path.exists(radar_file):
                rel_radar = os.path.relpath(radar_file, result_folder).replace(os.sep, '/')
                print(f"Radar relative path: {rel_radar}")
                print(f"Full URL would be: /results/{rel_radar}")
            
            if os.path.exists(analysis_file):
                rel_analysis = os.path.relpath(analysis_file, result_folder).replace(os.sep, '/')
                print(f"Analysis relative path: {rel_analysis}")
                print(f"Full URL would be: /results/{rel_analysis}")
            
            # Check associated uploads
            uploads = Upload.query.filter(
                (Upload.left_fencer_id == fencer.id) | (Upload.right_fencer_id == fencer.id)
            ).filter_by(status='completed').all()
            print(f"Associated uploads: {len(uploads)}")

if __name__ == "__main__":
    test_graph_paths()