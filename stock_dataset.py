import sqlite3
from typing import MutableMapping
import numpy as np

# 36526 ~ 44530까지의 시간을 가짐
class stock_dataset():
	'''
	float np로 정제된 dataset을 받아올 수 있는 클래스
	kospi, nasdaq, sp로 구성되어 있음.
	'''
	K = 1000
	M = 1000000
	B = 1000000000

	#filename을 받아 데이터베이스를 열게 됨.
	def init(self, filename):
		self.filename = filename


	#K / M / B을 변환해주는 함수
	def mulAbbreviation(self, floatAbb):
		floatReal = 0.0

		if 'K' in floatAbb:
			floatReal = float(floatAbb[0:-1]) * self.K
		elif 'M' in floatAbb:
			floatReal = float(floatAbb[0:-1]) * self.M
		elif 'B' in floatAbb:
			floatReal = float(floatAbb[0:-1]) * self.B
		return floatReal

	def kospi(self):
		#db 핸들러 받아옴
		conn = sqlite3.connect(self.filename)
		cur = conn.cursor()

		#SQL에 쿼리를 날려 Kospi 내의 모든 것을 받아옴
		sql = "select * from kospi order by date asc"
		cur.execute(sql)
		rows = cur.fetchall()

		#np 형태로 변환
		kospi_np = np.array(rows, dtype='float32')

		cur.close()
		conn.close()
		return kospi_np

	#위 함수와 다를게 없음
	def nasdaq(self):
		conn = sqlite3.connect(self.filename)
		cur = conn.cursor()

		sql = "select * from nasdaq order by date asc"

		cur.execute(sql)
		rows = cur.fetchall()

		for i in range(len(rows)):
			rows[i] = list(rows[i])
			rows[i][5] = self.mulAbbreviation(rows[i][5])

		nasdaq_np = np.array(rows, dtype='float32')

		cur.close()
		conn.close()
		return nasdaq_np

	def sp(self):
		conn = sqlite3.connect(self.filename)
		cur = conn.cursor()

		sql = "select * from sp order by date asc"

		cur.execute(sql)
		rows = cur.fetchall()

		for i in range(len(rows)):
			rows[i] = list(rows[i])
			rows[i][5] = self.mulAbbreviation(rows[i][5])

		sp_np = np.array(rows, dtype='float32')


		cur.close()
		conn.close()
		return sp_np

	def tech_indi(self):
		conn = sqlite3.connect(self.filename)
		cur = conn.cursor()

		kospi = "select date, stockper from kospi order by date asc"

		cur.execute(kospi)
		kospi_rows = np.array(cur.fetchall(), dtype='float32')

		nasdaq = "select percent * 100 from nasdaq order by date asc"

		cur.execute(nasdaq)
		nasdaq_rows = np.array(cur.fetchall(), dtype='float32')

		sp = "select percent * 100 from sp order by date asc"

		cur.execute(sp)
		sp_rows = np.array(cur.fetchall(), dtype='float32')

		tech_indi_np = np.concatenate((kospi_rows, nasdaq_rows, sp_rows), axis=1)

		return tech_indi_np



stock = stock_dataset()
stock.init("data.db")

print(stock.tech_indi())
