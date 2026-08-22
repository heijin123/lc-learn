import sqlite3
import os

db_path = os.path.join(os.path.dirname(__file__), "gold_data.db")
conn = sqlite3.connect(db_path)
cursor = conn.cursor()

cursor.execute("SELECT * FROM gold_data")
rows = cursor.fetchall()
for row in rows:
    print(row)