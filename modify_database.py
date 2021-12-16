import sqlite3 as sql
import time
from datetime import datetime
from dateutil.relativedelta import relativedelta
import numpy as np

def kospi(conn, cur):
	date = datetime.strptime('2000/01/01', '%Y/%m/%d')
	for i in range(7669):
		s = '{0}/{1:0>2}/{2:0>2}'.format(date.year, date.month, date.day)
		
		select = "select * from kospi where 일자 = ?"
		cur.execute(select, ("s", ))
		rows = cur.fetchall()

		for i in rows:
			print(i)

	conn.commit()


def kospiInit(conn, cur):
	addCol = "insert into kospi values(?, ?, ?, ?, ?)"
	cur.execute(addCol, ("2000/01/01", "1,059.04", "3.01", "195,899", "0.00"))
	cur.execute(addCol, ("2000/01/02", "1,059.04", "3.01", "195,899", "0.00"))
	cur.execute(addCol, ("2000/01/03", "1,059.04", "3.01", "195,899", "0.00"))

def nasdaqInit(conn,cur):
	addCol = "insert into nasdaq values(?, ?, ?, ?, ?, ?, ?)"
	cur.execute(addCol, ("2000년 01월 01일", "4,131.1", "4,186.2", "4,192.2", "3,989.7", "-", "1.52%"))
	cur.execute(addCol, ("2000년 01월 02일", "4,131.1", "4,186.2", "4,192.2", "3,989.7", "-", "1.52%"))

	conn.commit()

def spInit(conn, cur):
	addCol = "insert into sp values(?, ?, ?, ?, ?, ?, ?)"
	cur.execute(addCol, ("Jan 02, 2000", "145.44", "148.25", "148.25", "143.88", "8.16M", "-0.98%"))
	cur.execute(addCol, ("Jan 01, 2000", "145.44", "148.25", "148.25", "143.88", "8.16M", "-0.98%"))

	conn.commit()

if __name__=="__main__":
	conn = sql.connect("data.sqlite")

	cur = conn.cursor()

	#kospiInit(conn, cur)
	#nasdaqInit(conn, cur)
	#spInit(conn, cur)

	kospi(conn,cur)

	cur.close()
	conn.close()

