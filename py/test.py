import fsrs_optimizer_rust
import sqlite3

conn = sqlite3.connect("tests/data/collection.anki21")
c = conn.cursor()
c.execute("SELECT * FROM revlog")
values = c.fetchall()

COLUMNS = ["id", "cid", "usn", "ease", "ivl", "lastIvl", "factor", "time", "type"]

values = [{k: v for k,v in zip(COLUMNS, value)} for value in values]

print(values[0])

w = fsrs_optimizer_rust.py_train(values)

print(f"Python weights: {w}")