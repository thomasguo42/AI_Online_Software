#!/usr/bin/env python3
"""
Comprehensive check of VideoAnalysis records and Upload status
"""

import sys
import os
import json
from datetime import datetime

# Add the project directory to the Python path
sys.path.insert(0, '/workspace/Project')

from models import db, VideoAnalysis, Upload
from app import create_app

def check_video_analysis_records():
    """Check all VideoAnalysis records and their relationship to uploads"""
    
    app = create_app()
    
    with app.app_context():
        # Get all uploads with status 'completed'
        completed_uploads = Upload.query.filter_by(status='completed').order_by(Upload.id.desc()).all()
        
        print(f"📊 Found {len(completed_uploads)} completed uploads:")
        print("=" * 80)
        
        for upload in completed_uploads:
            print(f"🎯 Upload {upload.id} (User {upload.user_id})")
            print(f"   Upload Status: {upload.status}")
            print(f"   Created: {upload.created_at}")
            
            # Check if VideoAnalysis exists for this upload
            analysis = VideoAnalysis.query.filter_by(upload_id=upload.id).first()
            
            if analysis:
                print(f"   ✅ VideoAnalysis exists - Status: {analysis.status}")
                print(f"   📅 Generated: {analysis.generated_at}")
                
                # Check what data is available
                data_fields = []
                if analysis.left_overall_analysis:
                    try:
                        left_overall = json.loads(analysis.left_overall_analysis)
                        data_fields.append(f"left_overall({len(left_overall)} keys)")
                    except:
                        data_fields.append("left_overall(invalid JSON)")
                        
                if analysis.right_overall_analysis:
                    try:
                        right_overall = json.loads(analysis.right_overall_analysis)
                        data_fields.append(f"right_overall({len(right_overall)} keys)")
                    except:
                        data_fields.append("right_overall(invalid JSON)")
                        
                if analysis.left_category_analysis:
                    try:
                        left_category = json.loads(analysis.left_category_analysis)
                        data_fields.append(f"left_category({len(left_category)} categories)")
                    except:
                        data_fields.append("left_category(invalid JSON)")
                        
                if analysis.right_category_analysis:
                    try:
                        right_category = json.loads(analysis.right_category_analysis)
                        data_fields.append(f"right_category({len(right_category)} categories)")
                    except:
                        data_fields.append("right_category(invalid JSON)")
                        
                if analysis.loss_analysis:
                    try:
                        loss_data = json.loads(analysis.loss_analysis)
                        data_fields.append(f"loss_analysis({len(loss_data)} keys)")
                    except:
                        data_fields.append("loss_analysis(invalid JSON)")
                
                if data_fields:
                    print(f"   📊 Data: {', '.join(data_fields)}")
                else:
                    print("   ❌ No analysis data found")
                
                if hasattr(analysis, 'error_message') and analysis.error_message:
                    print(f"   🚨 Error: {analysis.error_message}")
                    
            else:
                print("   ❌ No VideoAnalysis record found")
                
                # Check if results directory exists
                result_dir = os.path.join('/workspace/Project/results', str(upload.user_id), str(upload.id))
                if os.path.exists(result_dir):
                    print(f"   📁 Results dir exists: {result_dir}")
                    
                    # Check for match_analysis directory
                    match_analysis_dir = os.path.join(result_dir, 'match_analysis')
                    if os.path.exists(match_analysis_dir):
                        analysis_files = [f for f in os.listdir(match_analysis_dir) if f.endswith('_analysis.json')]
                        print(f"   📄 Analysis files: {len(analysis_files)} found")
                    else:
                        print("   ❌ No match_analysis directory")
                else:
                    print("   ❌ No results directory")
            
            print("-" * 40)
        
        # Also check for orphaned VideoAnalysis records
        print("\n🔍 Checking for orphaned VideoAnalysis records...")
        all_analyses = VideoAnalysis.query.all()
        upload_ids = {upload.id for upload in completed_uploads}
        
        orphaned = [analysis for analysis in all_analyses if analysis.upload_id not in upload_ids]
        if orphaned:
            print(f"⚠️  Found {len(orphaned)} orphaned VideoAnalysis records:")
            for analysis in orphaned:
                print(f"   Upload {analysis.upload_id} - Status: {analysis.status}")
        else:
            print("✅ No orphaned VideoAnalysis records found")

if __name__ == "__main__":
    check_video_analysis_records()