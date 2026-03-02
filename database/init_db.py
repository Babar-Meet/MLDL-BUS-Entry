#!/usr/bin/env python3
"""
Database initialization script with proper path handling.
Creates database folder and initializes SQLite database.
"""

import os
import sys
import sqlite3

# Get the base directory (project root)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# Go up one level from database/ to project root
BASE_DIR = os.path.dirname(BASE_DIR)

def init_database():
    """Initialize database with proper path handling"""
    
    # Create database folder path
    db_folder = os.path.join(BASE_DIR, "database")
    
    # Ensure database folder exists BEFORE connecting
    try:
        os.makedirs(db_folder, exist_ok=True)
        print(f"[INFO] Database folder: {db_folder}")
    except Exception as e:
        print(f"[ERROR] Failed to create database folder: {e}")
        return False
    
    # Database file path
    db_path = os.path.join(db_folder, "vehicles.db")
    
    try:
        # Connect and create tables
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        # Create vehicles_log table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS vehicles_log (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                vehicle_number TEXT NOT NULL,
                vehicle_type TEXT NOT NULL,
                detection_date DATE NOT NULL,
                detection_time TIME NOT NULL,
                snapshot_path TEXT NOT NULL,
                confidence_score REAL NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        # Create indexes
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_vehicle_number ON vehicles_log(vehicle_number)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_detection_date ON vehicles_log(detection_date)')
        
        conn.commit()
        conn.close()
        
        print(f"[SUCCESS] Database initialized: {db_path}")
        return True
        
    except sqlite3.Error as e:
        print(f"[ERROR] SQLite error: {e}")
        return False
    except Exception as e:
        print(f"[ERROR] Unexpected error: {e}")
        return False


def test_database():
    """Test database connection"""
    db_path = os.path.join(BASE_DIR, "database", "vehicles.db")
    
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
        tables = cursor.fetchall()
        conn.close()
        
        print(f"[INFO] Database tables: {[t[0] for t in tables]}")
        return True
        
    except Exception as e:
        print(f"[ERROR] Database test failed: {e}")
        return False


if __name__ == "__main__":
    print("=" * 50)
    print("  DATABASE INITIALIZATION")
    print("=" * 50)
    print(f"Running from: {BASE_DIR}")
    print()
    
    success = init_database()
    
    if success:
        print()
        test_database()
        print()
        print("[SUCCESS] Database setup complete!")
        sys.exit(0)
    else:
        print()
        print("[ERROR] Database setup FAILED!")
        sys.exit(1)
