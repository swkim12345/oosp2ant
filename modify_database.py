import sqlite3
import time
from datetime import datetime
from dateutil.relativedelta import relativedelta
#import numpy as np

'''
이 함수는 처음에만 실행하게 됨.
더이상 실행할 이유 없음.
사용하지 마세요!
'''
#끝나는 날짜

# 36526 ~ 44530까지의 시간을 가짐

# def kospi(conn, cur):
# 	'''
# 	단순 노가다로 복사해주는 함수(date 의 unique가 보장되어 있으므로 이러한 방법을 사용함.)
# 	'''
# 	date_max = 36526
# 	date = 36529
# 	index_max = date_max - date
# 	for i in range(0, index_max, -1):
# 		select = "select * from kospi where date = ?"
# 		cur.execute(select, (str(date + i) + ".0", ))
# 		rows = cur.fetchall() #작동 잘 됨

# 		print(rows[0][0])

# 		insert_row = "insert or ignore into kospi (date, end, stockper, trading, percent) values(?, ?, ?, ?, ?)"
# 		cur.execute(insert_row, (str(date + i - 1) + ".0", rows[0][1], rows[0][2], rows[0][3], rows[0][4]))

# 	conn.commit()

# def nasdaq(conn, cur):
# 	#최대 : 44530, 최소 :36528
# 	date_max = 36526
# 	date = 36528
# 	index_max = date_max - date
# 	for i in range(0, index_max, -1):
# 		select = "select * from nasdaq where date = ?"
# 		cur.execute(select, (str(date + i) + ".0", ))
# 		rows = cur.fetchall() #작동 잘 됨

# 		print(rows[0][0])

# 		insert_row = "insert or ignore into nasdaq (date, end, open, high, low, frequency, percent) values(?, ?, ?, ?, ?, ?, ?)"
# 		cur.execute(insert_row, (str(date + i - 1) + ".0", rows[0][1], rows[0][2], rows[0][3], rows[0][4], rows[0][5], rows[0][6]))

# 	conn.commit()

def sp(conn, cur):
	date_max = 36526
	date = 36528
	index_max = date_max - date
	for i in range(0, index_max, -1):
		select = "select * from sp where date = ?"
		cur.execute(select, (str(date + i) + ".0", ))
		rows = cur.fetchall() #작동 잘 됨

		print(rows[0][0])

		insert_row = "insert or ignore into sp (date, end, open, high, low, trading, percent) values(?, ?, ?, ?, ?, ?, ?)"
		cur.execute(insert_row, (str(date + i - 1) + ".0", rows[0][1], rows[0][2], rows[0][3], rows[0][4], rows[0][5], rows[0][6]))

	conn.commit()

if __name__=="__main__":
	conn = sqlite3.connect("data.db")

	cur = conn.cursor()


	#kospi(conn,cur)

	#nasdaq(conn, cur)

	sp(conn, cur)

	# sql = "select date from kospi order by date desc"
	# cur.execute(sql)
	# rows = cur.fetchall()
	# print(rows)


	cur.close()
	conn.close()

