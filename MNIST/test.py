# -*- coding: UTF-8 -*-

import tensorflow as tf
import pandas as pd
import numpy as np

batch_size = 100
batch_index = 0

data = pd.read_csv('data/train.csv').sample(frac=1)
train_data = data[0:int(0.8 * len(data))]
verify_data = data[int(0.8 * len(data)):int(0.9 * len(data))]
test_data = data[int(0.9 * len(data)):-1]
train_y = train_data['label'].values
train_x = train_data.drop(labels=['label'], axis=1).values
verify_y = verify_data['label'].values
verify_x = verify_data.drop(labels=['label'], axis=1).values
test_y = test_data['label'].values
test_x = test_data.drop(labels=['label'], axis=1).values

del data, train_data, verify_data, test_data

train_x = train_x / 255
verify_x = verify_x / 255
test_x = test_x / 255


def next_batch():
	global batch_index
	batch_index += batch_size
	return train_y[batch_index - batch_size:batch_index], train_x[
	                                                      batch_index - batch_size:batch_index]


x = tf.placeholder(tf.float32, [None, 784])
y_ = tf.placeholder(tf.int64, [None])
w = tf.Variable(tf.zeros([784, 10]))
b = tf.Variable(tf.zeros([10]))

y = tf.matmul(x, w) + b

cross_entropy = tf.losses.sparse_softmax_cross_entropy(labels=y_, logits=y)

train_step = tf.train.GradientDescentOptimizer(0.01).minimize(cross_entropy)

init = tf.global_variables_initializer()

sess = tf.InteractiveSession()
sess.run(init)

correct_prediction = tf.equal(tf.argmax(y, 1), y_)
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

for epoch in range(30):
	for _ in range(int(len(train_x) / batch_size)):
		batch_y, batch_x = next_batch()
		sess.run(train_step, feed_dict={x: batch_x, y_: batch_y,})

	print(epoch, sess.run(
		accuracy, feed_dict={
			x: verify_x,
			y_: verify_y
		}))
	batch_index = 0

print(sess.run(
	accuracy, feed_dict={
		x: test_x,
		y_: test_y
	}))
