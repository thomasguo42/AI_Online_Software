#!/usr/bin/env python3
"""
Database migration script to add multi-video upload support.
Run this script to update the database schema.
"""

import sqlite3
import os
import sys

# Add the project root to the path so we can import our models
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def run_migration():
    """Run the database migration to add multi-video support (idempotent, SQLite-safe)"""

    # Database path
    db_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'instance', 'site.db')

    if not os.path.exists(db_path):
        print(f"Database not found at {db_path}")
        return False

    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    def column_exists(table: str, column: str) -> bool:
        try:
            cursor.execute(f"PRAGMA table_info({table})")
            cols = [c[1] for c in cursor.fetchall()]
            return column in cols
        except Exception:
            return False

    def table_exists(table: str) -> bool:
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name=?", (table,))
        return cursor.fetchone() is not None

    try:
        print("Starting multi-video upload migration...")

        # 1) Ensure columns on upload
        try:
            if not column_exists('upload', 'is_multi_video'):
                print("Adding column upload.is_multi_video ...")
                cursor.execute("ALTER TABLE upload ADD COLUMN is_multi_video BOOLEAN DEFAULT 0")
            if not column_exists('upload', 'match_title'):
                print("Adding column upload.match_title ...")
                cursor.execute("ALTER TABLE upload ADD COLUMN match_title VARCHAR(255)")
            conn.commit()
        except Exception as e:
            print(f"Warning: could not add columns to upload: {e}")
            conn.commit()

        # 2) Ensure upload_video table exists
        try:
            if not table_exists('upload_video'):
                print("Creating table upload_video ...")
                cursor.execute(
                    """
                    CREATE TABLE IF NOT EXISTS upload_video (
                        id INTEGER PRIMARY KEY,
                        upload_id INTEGER NOT NULL,
                        video_path VARCHAR(255) NOT NULL,
                        sequence_order INTEGER NOT NULL,
                        selected_indexes VARCHAR(50),
                        status VARCHAR(50) DEFAULT 'pending',
                        detection_image_path VARCHAR(255),
                        total_bouts INTEGER DEFAULT 0,
                        bouts_offset INTEGER DEFAULT 0,
                        FOREIGN KEY (upload_id) REFERENCES upload (id)
                    )
                    """
                )
            conn.commit()
        except Exception as e:
            print(f"Warning: could not create upload_video table: {e}")
            conn.commit()

        # 3) Ensure bout.upload_video_id column exists (skip adding FK constraint via ALTER in SQLite)
        try:
            if not column_exists('bout', 'upload_video_id'):
                print("Adding column bout.upload_video_id ...")
                cursor.execute("ALTER TABLE bout ADD COLUMN upload_video_id INTEGER")
            conn.commit()
        except Exception as e:
            print(f"Warning: could not add column to bout: {e}")
            conn.commit()

        # 4) Migrate existing uploads to upload_video rows if not already
        try:
            print("Migrating existing uploads -> upload_video ...")
            cursor.execute("SELECT id, video_path, selected_indexes, detection_image_path, total_bouts FROM upload WHERE video_path IS NOT NULL")
            existing_uploads = cursor.fetchall()
            for upload_id, video_path, selected_indexes, detection_image_path, total_bouts in existing_uploads:
                # Check if an upload_video already exists for this upload
                cursor.execute("SELECT id FROM upload_video WHERE upload_id = ?", (upload_id,))
                if cursor.fetchone() is None:
                    cursor.execute(
                        """
                        INSERT INTO upload_video 
                        (upload_id, video_path, sequence_order, selected_indexes, status, detection_image_path, total_bouts, bouts_offset)
                        VALUES (?, ?, 1, ?, 'completed', ?, ?, 0)
                        """,
                        (upload_id, video_path or '', selected_indexes or '', detection_image_path or '', total_bouts or 0)
                    )
                    upload_video_id = cursor.lastrowid
                    cursor.execute("UPDATE bout SET upload_video_id = ? WHERE upload_id = ?", (upload_video_id, upload_id))
            conn.commit()
        except Exception as e:
            print(f"Warning: migration of existing uploads failed: {e}")
            conn.commit()

        # 5) Indexes (IF NOT EXISTS)
        try:
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_upload_video_upload_id ON upload_video(upload_id)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_upload_video_sequence ON upload_video(upload_id, sequence_order)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_bout_upload_video_id ON bout(upload_video_id)")
            conn.commit()
        except Exception as e:
            print(f"Warning: could not create indexes: {e}")
            conn.commit()

        print("Migration completed (idempotent).")
        return True

    except Exception as e:
        print(f"Migration failed fatally: {e}")
        return False
    finally:
        conn.close()

if __name__ == "__main__":
    if run_migration():
        print("✅ Multi-video upload support added successfully!")
    else:
        print("❌ Migration failed!")
        sys.exit(1)