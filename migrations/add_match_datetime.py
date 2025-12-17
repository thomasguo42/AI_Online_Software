#!/usr/bin/env python3
"""Add match_datetime column to upload table (idempotent, SQLite-safe)."""

import os
import sqlite3
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def run_migration():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    db_path = os.path.join(base_dir, 'instance', 'site.db')

    if not os.path.exists(db_path):
        print(f"Database not found at {db_path}")
        return False

    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    def column_exists(table: str, column: str) -> bool:
        try:
            cursor.execute(f"PRAGMA table_info({table})")
            return any(col[1] == column for col in cursor.fetchall())
        except Exception:
            return False

    try:
        if not column_exists('upload', 'match_datetime'):
            print('Adding column upload.match_datetime ...')
            cursor.execute("ALTER TABLE upload ADD COLUMN match_datetime TIMESTAMP")
            conn.commit()
        else:
            print('Column upload.match_datetime already exists; skipping.')
        return True
    except Exception as exc:
        print(f"Failed to add upload.match_datetime: {exc}")
        conn.rollback()
        return False
    finally:
        conn.close()

if __name__ == '__main__':
    if run_migration():
        print('✅ match_datetime migration completed successfully!')
    else:
        print('❌ match_datetime migration failed!')
        sys.exit(1)
