#!/usr/bin/env python3
"""
Fix script to assign fencers to upload 101 so we can test the profile graphs.
"""

import sys
sys.path.append('/workspace/Project')

from app import create_app
from models import db, User, Fencer, Upload

def fix_upload_101():
    """Fix upload 101 by assigning test fencers to it."""
    print("Fixing Upload 101 Fencer Assignments")
    print("=" * 50)
    
    app = create_app()
    
    with app.app_context():
        # Get upload 101
        upload = Upload.query.get(101)
        if not upload:
            print("❌ Upload 101 not found!")
            return
        
        print(f"Upload 101 found: User {upload.user_id}")
        print(f"Current assignments - Left: {upload.left_fencer_id}, Right: {upload.right_fencer_id}")
        
        # Get or create test fencers for user 1
        user_1 = User.query.get(1)
        if not user_1:
            print("❌ User 1 not found!")
            return
        
        # Check if we have fencers for user 1
        user_1_fencers = Fencer.query.filter_by(user_id=1).all()
        print(f"User 1 has {len(user_1_fencers)} fencers:")
        for f in user_1_fencers:
            print(f"  - Fencer {f.id}: {f.name}")
        
        if len(user_1_fencers) >= 2:
            # Use existing fencers
            left_fencer = user_1_fencers[0]
            right_fencer = user_1_fencers[1]
            
            print(f"\nAssigning existing fencers:")
            print(f"  Left: {left_fencer.name} (ID: {left_fencer.id})")
            print(f"  Right: {right_fencer.name} (ID: {right_fencer.id})")
        else:
            # Create test fencers if needed
            print(f"\nCreating test fencers...")
            
            if len(user_1_fencers) == 0:
                left_fencer = Fencer(name="Test Left", user_id=1)
                db.session.add(left_fencer)
                
                right_fencer = Fencer(name="Test Right", user_id=1)
                db.session.add(right_fencer)
            else:
                left_fencer = user_1_fencers[0]
                right_fencer = Fencer(name="Test Right", user_id=1)
                db.session.add(right_fencer)
            
            db.session.commit()  # Commit to get IDs
            
            print(f"  Left: {left_fencer.name} (ID: {left_fencer.id})")
            print(f"  Right: {right_fencer.name} (ID: {right_fencer.id})")
        
        # Update upload 101
        upload.left_fencer_id = left_fencer.id
        upload.right_fencer_id = right_fencer.id
        
        db.session.commit()
        
        print(f"\n✅ Upload 101 updated successfully!")
        print(f"   Left Fencer: {left_fencer.name} (ID: {left_fencer.id})")
        print(f"   Right Fencer: {right_fencer.name} (ID: {right_fencer.id})")
        
        print(f"\n🎯 Now you can test by:")
        print(f"   1. Login as user '1234' (user ID 1)")
        print(f"   2. Go to Fencer Management")
        print(f"   3. Click 'View Profile' for '{left_fencer.name}' or '{right_fencer.name}'")
        print(f"   4. You should see the profile graphs!")

if __name__ == "__main__":
    fix_upload_101()