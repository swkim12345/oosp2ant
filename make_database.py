import sqlite3
import os

#make_database는 기본적으로 데이터베이스에 넣는 것에 중점을 둔다
#이외의 수정은 modify_database에서 진행한다.
class make_database:
	#데이터 베이스 생성자
	def __init__(self, date_format, skema):
		#윈도우, 맥환경에서 둘 다 실행하기 위한 툴임.
		import platform

		if platform.system() == 'Windows':
			self.platform = 'Windows'
		elif platform.system() == 'Darwin':
			self.platform = 'Darwin'
		else:
			self.platform = 'N'
			self.__del__()

		self.date_format = date_format
		self.skema = skema

	#소멸자
	def __del__(self):
		try :
			pass
		except self.platform == 'N':
			print("운영체제가 잘못되었습니다.")

	#스키마 :
	def make_database(self):
		if self.platform == 'Window':
			self.database_location = os.
		con = sqlite3.connect('dataset.sqlite')

	#데이터 로드
	def load_rawdata():
		pass

	#데이터 규칙 생성(휴장일, 토, 일요일은 전, 혹은 전전전전저....일의 값을 복사해 오는 것으로)
	def make_data_rule():
		pass

	#데이터베이스에 입력
	def inject_database():
		pass

if '__name__' == '__main__' :
	pass
