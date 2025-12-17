#!/usr/bin/env python3
"""
Fix VideoAnalysis records by removing failed/error records to allow regeneration
"""

import sys
import os
from datetime import datetime

# Add the project directory to the Python path
sys.path.insert(0, '/workspace/Project')

from models import db, VideoAnalysis, Upload
from app import create_app

def fix_video_analysis_records():
    """Remove failed VideoAnalysis records to allow regeneration"""
    
    app = create_app()
    
    with app.app_context():
        # Find all VideoAnalysis records with error status
        error_analyses = VideoAnalysis.query.filter_by(status='error').all()
        
        if not error_analyses:
            print("✅ No error VideoAnalysis records found")
            return
        
        print(f"🔍 Found {len(error_analyses)} VideoAnalysis records with error status:")
        print("=" * 60)
        
        deleted_count = 0
        for analysis in error_analyses:
            # Get upload info
            upload = Upload.query.get(analysis.upload_id)
            upload_info = f"Upload {analysis.upload_id}"
            if upload:
                upload_info += f" (User {upload.user_id}, Upload Status: {upload.status})"
            
            print(f"❌ {upload_info}")
            print(f"   Analysis Status: {analysis.status}")
            print(f"   Generated: {analysis.generated_at}")
            if hasattr(analysis, 'error_message') and analysis.error_message:
                print(f"   Error: {analysis.error_message}")
            
            # Check if upload is completed (can be regenerated)
            if upload and upload.status == 'completed':
                print("   🔄 Upload is completed - can regenerate analysis")
                
                # Delete the failed record
                db.session.delete(analysis)
                deleted_count += 1
                print("   🗑️  Deleted failed VideoAnalysis record")
            else:
                print("   ⚠️  Upload not completed - keeping error record")
            
            print("-" * 40)
        
        # Commit changes
        try:
            db.session.commit()
            print(f"\n✅ Successfully cleaned up {deleted_count} failed VideoAnalysis records")
            print("💡 Next time you visit the video view page, it will regenerate the analysis")
            
            # Also trigger immediate regeneration for the most recent upload
            if deleted_count > 0:
                recent_upload = Upload.query.filter_by(status='completed').order_by(Upload.id.desc()).first()
                if recent_upload:
                    print(f"\n🔄 Triggering regeneration for most recent upload {recent_upload.id}")
                    trigger_analysis_regeneration(recent_upload.id)
                    
        except Exception as e:
            db.session.rollback()
            print(f"\n❌ Error committing changes: {e}")

def trigger_analysis_regeneration(upload_id):
    """Trigger immediate regeneration of analysis for an upload"""
    try:
        import os
        from app import create_app
        
        app = create_app()
        with app.app_context():
            # Import the analysis function
            from tasks import _generate_ai_analysis_for_upload
            
            # Get upload and check if results exist
            upload = Upload.query.get(upload_id)
            if not upload:
                print(f"❌ Upload {upload_id} not found")
                return
                
            result_dir = os.path.join('/workspace/Project/results', str(upload.user_id), str(upload_id))
            if not os.path.exists(result_dir):
                print(f"❌ Results directory not found: {result_dir}")
                return
                
            print(f"🚀 Starting AI analysis regeneration for upload {upload_id}")
            _generate_ai_analysis_for_upload(upload_id, result_dir, db)
            print(f"✅ Analysis regeneration completed for upload {upload_id}")
            
    except Exception as e:
        print(f"❌ Error during analysis regeneration: {e}")
        import traceback
        print(traceback.format_exc())

if __name__ == "__main__":
    fix_video_analysis_records()