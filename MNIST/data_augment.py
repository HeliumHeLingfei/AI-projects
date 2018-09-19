# -*- coding: UTF-8 -*-
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# ldata = pd.read_csv('data/train.csv')
# data = ldata.values[:,1:]
# label = ldata.values[:,0]
# print(label)
# print(data)
# data.dump('data/data0')
# label.dump('data/label')
# del ldata
#
# data1=data
# zero=np.zeros([42000,],dtype=np.int32)
# image=np.reshape(data1[0],[28,28])
# plt.imshow(image,cmap='gray')
# plt.show()
# for i in range(28):
# 	data1=np.insert(data1,i*28+28,values=zero,axis=1)
# 	data1=np.insert(data1,i*28+28,values=zero,axis=1)
# 	data1=np.insert(data1,i*28+28,values=zero,axis=1)
# 	data1=np.delete(data1,[i*28,i*28+1,i*28+2],axis=1)
# image=np.reshape(data1[0],[28,28])
# plt.imshow(image,cmap='gray')
# plt.show()
# data1.dump('data/data1')
#
# data2=data
# for i in range(28):
# 	data2=np.insert(data2,i*28,values=zero,axis=1)
# 	data2=np.insert(data2,i*28,values=zero,axis=1)
# 	data2=np.insert(data2,i*28,values=zero,axis=1)
# 	data2=np.delete(data2,[(i+1)*28,(i+1)*28+1,(i+1)*28+2],axis=1)
# image=np.reshape(data2[0],[28,28])
# plt.imshow(image,cmap='gray')
# plt.show()
# data2.dump('data/data2')
#
# data1 = data
# zero = np.zeros([42000, ], dtype=np.int32)
# image = np.reshape(data1[0], [28, 28])
# print(image, '\n')
# plt.imshow(image, cmap='gray')
# plt.show()
# for i in range(0, 84):
# 	data1 = np.delete(data1, 0, axis=1)
# image = np.reshape(data1[0], [25, 28])
# print(image)
# for i in range(700, 784):
# 	data1 = np.insert(data1, i, values=zero, axis=1)
# image = np.reshape(data1[0], [28, 28])
# print(image)
# plt.imshow(image, cmap='gray')
# plt.show()
# data1.dump('data/data3')
#
# data1 = data
# zero = np.zeros([42000, ], dtype=np.int32)
# image = np.reshape(data1[0], [28, 28])
# print(image, '\n')
# plt.imshow(image, cmap='gray')
# plt.show()
# for i in range(700, 784):
# 	data1 = np.delete(data1, 700, axis=1)
# image = np.reshape(data1[0], [25, 28])
# print(image)
# for i in range(0, 84):
# 	data1 = np.insert(data1, 0,values=zero, axis=1)
# image = np.reshape(data1[0], [28, 28])
# print(image)
# plt.imshow(image, cmap='gray')
# plt.show()
# data1.dump('data/data4')


data = np.load('data/data0')
llabel = np.load('data/label')
print(data.shape)
print(llabel.shape)
label = llabel
for i in range(1, 5):
	ldata = np.load('data/data' + str(i))
	data = np.vstack((data, ldata))
	label = np.hstack((label, llabel))
label=np.transpose([label])
data = np.hstack((label, data))
print(data.shape)
print(data)
data.dump('data/all_data')
