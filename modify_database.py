import sqlite3 as sql
#import numpy as np

conn = sql.connect("data.sqlite")

cur = conn.cursor()

cur.execute("select * from kospi")
rows = cur.fetchall()

for i in rows:
	print(i)

conn.commit()
cur.close()
