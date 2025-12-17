#!/usr/bin/env python3
"""
Migration script to add extended_video_path column to Bout table
This column will store paths to extended videos with 1s padding for display purposes
"""

import os
import sys
from sqlalchemy import create_engine, text
from sqlalchemy.exc import OperationalError

def get_database_url():
    """Get database URL from environment or use default SQLite"""
    return os.environ.get('DATABASE_URL', 'sqlite:///fencing_analysis.db')

def migrate_database():
    """Add extended_video_path column to Bout table"""
    database_url = get_database_url()
    engine = create_engine(database_url)
    
    try:
        with engine.connect() as conn:
            # Check if bout table exists
            try:
                conn.execute(text("SELECT name FROM sqlite_master WHERE type='table' AND name='bout'"))
                table_exists = conn.fetchone() is not None
            except:
                table_exists = False
            
            if not table_exists:
                print("Bout table doesn't exist yet. This is normal for new installations.")
                print("The extended_video_path column will be created when the database is initialized.")
                return True
            
            # Check if column already exists
            try:
                result = conn.execute(text("SELECT extended_video_path FROM bout LIMIT 1"))
                print("Column 'extended_video_path' already exists in bout table")
                return True
            except OperationalError:
                # Column doesn't exist, add it
                pass
            
            # Add the new column
            conn.execute(text("ALTER TABLE bout ADD COLUMN extended_video_path VARCHAR(255)"))
            conn.commit()
            print("Successfully added 'extended_video_path' column to bout table")
            return True
            
    except Exception as e:
        print(f"Error during migration: {str(e)}")
        return False

if __name__ == "__main__":
    print("Running database migration to add extended_video_path column...")
    success = migrate_database()
    if success:
        print("Migration completed successfully!")
        sys.exit(0)
    else:
        print("Migration failed!")
        sys.exit(1) 