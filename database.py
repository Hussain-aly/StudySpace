import sqlite3

DB_NAME = "study_data.db"

def init_db():
    """Creates the database file and the subjects table, if they don't already exist."""
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS subjects (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            notes_text TEXT NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)
    conn.commit()
    conn.close()

def add_subject(name, notes_text):
    """Saves a new subject with its notes into the database."""
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    cursor.execute(
        "INSERT INTO subjects (name, notes_text) VALUES (?, ?)",
        (name, notes_text)
    )
    conn.commit()
    conn.close()

def get_all_subjects():
    """Returns a list of all saved subjects, as (id, name) pairs."""
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    cursor.execute("SELECT id, name FROM subjects ORDER BY created_at DESC")
    results = cursor.fetchall()
    conn.close()
    return results

def get_subject(subject_id):
    """Returns the full details of one subject, by its id."""
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    cursor.execute("SELECT id, name, notes_text FROM subjects WHERE id = ?", (subject_id,))
    result = cursor.fetchone()
    conn.close()
    return result

def delete_subject(subject_id):
    """Permanently removes a subject from the database, by its id."""
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    cursor.execute("DELETE FROM subjects WHERE id = ?", (subject_id,))
    conn.commit()
    conn.close()