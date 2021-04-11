import csv
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split


def load_data(filename):
	'''读取保存在csv文件中的时间序列模型数据，并对其进行标准化处理.'''
	try:
		with open(filename) as csvfile:
			reader = csv.reader(csvfile)
			data = [float(row[1]) for row in reader]
			nomalized = (data - np.mean(data,axis=0)) / np.std(data,axis=0)
		return nomalized
	except IOError:	#检测读取文件时是否有异常
		print("There is an IOError!")
		return

def split_data(data,split=0.8):
	'''训练集和测试集的划分，默认值为8:2.'''
	train_len = len(data)*split
	train,test = [],[]
	for index,row in enumerate(data):
		if index < train_len:
			train.append(row)
		else:
			test.append(row)
	return train,test
