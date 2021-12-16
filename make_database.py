import os
import platform
import csv
import pymysql

#안 쓰이는 함수임. 쓰지 마세요... 제발
'''
.sqlite3 (데이터 베이스명)
.create table (테이블명){}
.separator ','
.import (csv 파일) (데이터베이스명)
'''


#os 확인 후 경로 설정
def set_file_locate(file_name):
	if platform.system() == 'Windows':
		database_location = os.path.dirname(os.path.realpath(__file__)) + "\\" + file_name
	elif platform.system() == 'Darwin':
		database_location = os.path.dirname(os.path.realpath(__file__))  + "/" + file_name
	return database_location

if __name__ == '__main__' :
	#os 확인
	database_location = set_file_locate('dataset.sqlite')

	#database 생성 후 cursor 가져옴
	print(database_location)
	con = pymysql.connect(host='127.0.0.1', user='root', password='12345678', db='ossp_ant', charset='utf8')
	curs = con.cursor()

	#create table
	create_kospi_table = "CREATE TABLE kospi (DATE varchar(80) NOT NULL, CLOSING_PRICE varchar(80) NOT NULL, DAY_RANGE varchar(80) NOT NULL, TRADING_RATE varchar(80) NOT NULL, TRADING_RANGE varchar(80) NOT NULL)"


	# curs.execute("if exists DROP table kospi")
	# curs.execute(create_kospi_table)

	#dataload 2000~2021 kospi.csv
	kospi_f = open(set_file_locate('kospi.csv'), 'r', encoding='utf-8')


	for i in kospi_f :
		print(i)
		curs.execute("INSERT INTO kospi (DATE, CLOSING_PRICE, DAY_RANGE, TRADING_RATE, TRADING_RANGE) values(%s, %s, %s, %s, %s)", (i[0], i[1], i[2], i[3], i[4]))


