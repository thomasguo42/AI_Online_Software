#!/usr/bin/env python3

import os
import sys

# Add the project directory to Python path
sys.path.insert(0, '/workspace/Project')

from models import db, Upload
from app import create_app
from sqlalchemy import text

def main():
    """Create database migration for weapon_type column"""
    app = create_app()
    
    with app.app_context():
        try:
            # Check if weapon_type column exists
            result = db.session.execute(text('PRAGMA table_info(upload)'))
            columns = [row[1] for row in result.fetchall()]
            
            if 'weapon_type' in columns:
                print("✓ weapon_type column already exists")
                return
            
            print("⚠ weapon_type column does not exist, creating it...")
            
            # Add the weapon_type column with default value 'saber'
            db.session.execute(text('ALTER TABLE upload ADD COLUMN weapon_type VARCHAR(20) DEFAULT "saber"'))
            db.session.commit()
            
            print("✓ Successfully added weapon_type column")
            
            # Update all existing uploads to have weapon_type='saber' 
            result = db.session.execute(text('UPDATE upload SET weapon_type = "saber" WHERE weapon_type IS NULL'))
            db.session.commit()
            
            uploads_updated = result.rowcount
            print(f"✓ Updated {uploads_updated} existing uploads with default weapon_type='saber'")
            
        except Exception as e:
            print(f"❌ Error during migration: {e}")
            db.session.rollback()
            raise

if __name__ == "__main__":
    main()