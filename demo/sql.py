import sqlite3
import os

DB_PATH = r"F:\workspace\lc-learn\demo\person.db"

parent_dir = os.path.dirname(DB_PATH)
if not os.path.exists(parent_dir):
    os.makedirs(parent_dir)

def create_db():
   sql = """
   CREATE TABLE IF NOT EXISTS person (
       id INTEGER PRIMARY KEY AUTOINCREMENT,
       name TEXT,
       age INTEGER,
       weight TEXT,
       height TEXT
   );
   """
   with sqlite3.connect(DB_PATH) as conn:
       conn = conn.cursor()
       conn.execute(sql)
       
def insert_person(name: str, age: int, weight: str, height: str):
   sql = """
   INSERT INTO person (name, age, weight, height)
   VALUES (?, ?, ?, ?)
   """
   with sqlite3.connect(DB_PATH) as conn:
       conn = conn.cursor()
       conn.execute(sql, (name, age, weight, height))
def select_all_person():
   sql = """
   SELECT * FROM person
   """
   with sqlite3.connect(DB_PATH) as conn:
       conn = conn.cursor()
       rows = conn.execute(sql).fetchall()
       return rows
def update_person(id: int, name: str, age: int, weight: str, height: str):
   sql = """
   UPDATE person
   SET name = ?, age = ?, weight = ?, height = ?
   WHERE id = ?
   """
   with sqlite3.connect(DB_PATH) as conn:
       conn = conn.cursor()
       conn.execute(sql, (name, age, weight, height, id))
def delete_person(id: int):
   sql = """
   DELETE FROM person
   WHERE id = ?
   """
   with sqlite3.connect(DB_PATH) as conn:
       conn = conn.cursor()
       conn.execute(sql, (id,))

if __name__ == "__main__":
   print(select_all_person())