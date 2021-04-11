import tensorflow as tf
from tensorflow.contrib import rnn
import numpy as np
import matplotlib.pyplot as plt
import math

import data_load


class Predictor:
	'''实现时间序列模型训练和测试操作的类.'''

	def __init__(self,in_size=1,seq_size=5,hidden_size=100,learning=0.001,iteration=10000):
		'''模型各参数初始化，定义损失函数.'''
		self.in_size = in_size
		self.seq_size = seq_size
		self.hidden_size = hidden_size
		self.learning = learning
		self.iteration = iteration

		self.weight = tf.Variable(tf.random.normal([hidden_size,1],name = 'weight'))
		self.b = tf.Variable(tf.random.normal([1]),name = 'b')
		self.x = tf.compat.v1.placeholder(tf.float32,[None,seq_size,in_size])
		self.y = tf.compat.v1.placeholder(tf.float32,[None,seq_size])

		self.loss = tf.reduce_mean(tf.square(self.out_data() - self.y))
		self.opt = tf.train.AdamOptimizer(learning_rate=0.0001).minimize(self.loss)
		self.mse = 0

		self.saver = tf.compat.v1.train.Saver()

	def out_data(self):
		'''基础LSTM循环网络单元的输出.'''
		cell = rnn.BasicLSTMCell(self.hidden_size)
		output,state = tf.nn.dynamic_rnn(cell,self.x,dtype=tf.float32)
		
		num = tf.shape(self.x)[0]
		w = tf.tile(tf.expand_dims(self.weight,0),[num,1,1])
		out = tf.matmul(output,w) + self.b
		out = tf.squeeze(out)

		return out

	def train(self,train_x,train_y,test_x,test_y):
		'''训练模块，得到各步训练集误差和测试集误差，最终训练完成后保存当前网络参数模型.'''
		with tf.compat.v1.Session() as sess:
			tf.compat.v1.get_variable_scope().reuse_variables()	#重复利用RNN参数
			sess.run(tf.compat.v1.global_variables_initializer())

			step = 0
			for _ in range(self.iteration):
				_,train_err = sess.run([self.opt,self.loss],feed_dict={self.x:train_x,self.y:train_y})	#训练集误差
				test_err = sess.run(self.loss,feed_dict={self.x:test_x,self.y:test_y})	#测试集误差
				if step % 500 == 0:
					print('step: {}\t\ttrain err: {}\t\ttest err: {}'.format(step, train_err, test_err))
				step +=1
				self.mse += test_err

			save_path = self.saver.save(sess,'./model/')	#保存模型
			print('Model is saved to {}'.format(save_path))
		#返回训练后的模型性能指标
		return self.mse

	def test(self,sess,test_x):
		'''测试模块，读取已训练好的网络参数模型，最终得到当前预测值.'''
		tf.get_variable_scope().reuse_variables()
		self.saver.restore(sess,'./model/')	#读取模型

		return sess.run(self.out_data(),feed_dict={self.x:test_x})
	

def plot_show(train,pre,real,filename):
	'''数据可视化函数.'''
	plt.figure()

	num = len(train)
	plt.plot(list(range(num)),train,color='k',label='train data')
	plt.plot(list(range(num,num+len(pre))),pre,color='r',label='predicted data')
	plt.plot(list(range(num,num+len(real))),real,color='c',label='test data')

	plt.legend(title='The prediction of COVID19')
	if filename is not None:
		plt.savefig(filename)
	else:
		plt.show()


if __name__=='__main__':
	'''超参数在此取值，进行训练'''
	para = {'in_size':1,'seq_size':7,'hidden_size':100,'learning':0.0001,'iteration':10000}
	split = 0.5
	preidict = Predictor(para.get('in_size'),para.get('seq_size'),para.get('hidden_size'),para.get('learning'),para.get('iteration'))
	data = data_load.load_data('recovered.csv')

	sumMSE = []
	tmp = 0
	while split<=0.9:
		'''改变训练数据与测试数据的取值范围，进行交叉验证.'''
		train_data,test_data = data_load.split_data(data,split)
		print(train_data)
		print(test_data)

	
		train_x,train_y = [],[]
		#以一天为单位，每个train_y都和上一个train_x正好错开
		for i in range(len(train_data) - para.get('seq_size')):
			train_x.append(np.expand_dims(train_data[i:i+para.get('seq_size')],axis=1).tolist())
			train_y.append(train_data[i+1:i+para.get('seq_size')+1])

		test_x,test_y = [],[]
		for i in range(len(test_data)-para.get('seq_size')):
			test_x.append(np.expand_dims(test_data[i:i+para.get('seq_size')],axis=1).tolist())
			test_y.append(test_data[i+1:i+para.get('seq_size')+1])

		#返回得到整个训练集的损失和
		nowMSE = preidict.train(train_x,train_y,test_x,test_y)
		sumMSE.append(nowMSE)
		print('The data is divided into train data:' + "{:.12g}".format(split) + ' and test data:' + "{:.12g}".format(1-split))
		print('The MSE of this model is ' + str(sumMSE[tmp]/len(test_data) if tmp==0 else (nowMSE-sumMSE[tmp-1])/len(test_data)))

		tmp +=1
		split += 0.1

	with tf.compat.v1.Session() as sess:
			#数据切片时步长为1，得到预测结果
			preidict_data = preidict.test(sess,test_x)[:,0]
			plot_show(train_data,preidict_data,test_data,'prediction.png')




