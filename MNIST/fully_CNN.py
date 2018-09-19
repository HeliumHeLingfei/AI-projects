# -*- coding: UTF-8 -*-

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.contrib import layers

BATCH_SIZE = 100
NUM_EPOCHS = 30
IMAGE_LENGTH = 28
IMAGE_SIZE = IMAGE_LENGTH * IMAGE_LENGTH
KERNEL_SIZE = 5
POOLING_SIZE = 2
FIRST_FILTER = 32
SECOND_FILTER = 64
DENSE_UNITS = 256
CONV_ACTIVATION = tf.nn.relu
DENSE_ACTIVATION = tf.nn.relu
DENSE_REGULARIZER = None  # layers.l2_regularizer(0.0001)
LOSS_REGULARIZER = None  # layers.l2_regularizer(0.0001)



def cnn_model(image, labels, dropout_rate):
	def conv2d(inputs, filters):
		return tf.layers.conv2d(inputs=inputs,
		                        filters=filters,
		                        kernel_size=[KERNEL_SIZE, KERNEL_SIZE],
		                        padding="SAME",
		                        activation=CONV_ACTIVATION)

	def max_pool(input):
		return tf.nn.max_pool(value=input, ksize=[1, POOLING_SIZE, POOLING_SIZE, 1],
		                      strides=[1, POOLING_SIZE, POOLING_SIZE, 1], padding="SAME")

	def dense(inputs, units, activation, regularizer):
		return tf.layers.dense(
			inputs=inputs, units=units, activation=activation, bias_regularizer=regularizer)

	def dropout(inputs, dropout_rate):
		return tf.layers.dropout(
			inputs=inputs, rate=dropout_rate)

	with tf.name_scope('reshape'):
		x = tf.reshape(image, [-1, IMAGE_LENGTH, IMAGE_LENGTH, 1])

	with tf.name_scope('conv1'):
		conv1 = conv2d(x, FIRST_FILTER)

	with tf.name_scope('conv2'):
		conv2 = conv2d(conv1, FIRST_FILTER)

	with tf.name_scope('pool1'):
		pool1 = max_pool(conv2)

	with tf.name_scope('cdrop1'):
		cdrop1 = dropout(pool1, dropout_rate / 2)

	with tf.name_scope('conv3'):
		conv3 = conv2d(cdrop1, SECOND_FILTER)

	with tf.name_scope('conv4'):
		conv4 = conv2d(conv3, SECOND_FILTER)

	with tf.name_scope('pool2'):
		pool2 = max_pool(conv4)

	with tf.name_scope('cdrop2'):
		cdrop2 = dropout(pool2, dropout_rate / 2)

	with tf.name_scope('fully_connect'):
		flat = tf.reshape(cdrop2, [-1, 7 * 7 * SECOND_FILTER])
		dense1 = dense(flat, DENSE_UNITS, DENSE_ACTIVATION, DENSE_REGULARIZER)

	with tf.name_scope('dropout'):
		drop = dropout(dense1, dropout_rate)

	with tf.name_scope('logits'):
		weights = tf.Variable(tf.truncated_normal([DENSE_UNITS, 10], stddev=0.1, dtype=tf.float32))
		bias = tf.Variable(tf.constant(0.1, shape=[10], dtype=tf.float32))
		logits = tf.matmul(drop, weights) + bias

	with tf.name_scope('loss'):
		global_step=tf.Variable(0)
		cross_entropy = tf.losses.sparse_softmax_cross_entropy(labels=labels, logits=logits)
		loss = tf.reduce_mean(cross_entropy) #+ LOSS_REGULARIZER(weights)

	with tf.name_scope('optimizer'):
		learning_rate = tf.train.exponential_decay(1e-4, global_step, decay_steps=100, decay_rate=0.96,
		                                           staircase=True)
		optimizer = tf.train.AdamOptimizer(learning_rate)
		train_step = optimizer.minimize(loss,global_step=global_step)

	with tf.name_scope('accuracy'):
		correct_prediction = tf.cast(tf.equal(tf.argmax(logits, 1), labels), tf.float64)
		accuracy = tf.reduce_mean(correct_prediction)

	return train_step, accuracy, learning_rate, logits


def train(_):
	print("train:\n")
	data = np.load('data/all_data')
	np.random.shuffle(data)
	train_data = data[0:int(0.95 * len(data))]
	verify_data = data[int(0.95 * len(data)):-1]
	train_y = train_data[:,0]
	train_x = train_data[:,1:]
	verify_y = verify_data[:,0]
	verify_x = verify_data[:,1:]

	del data, train_data, verify_data

	train_x = train_x / 255
	verify_x = verify_x / 255

	x = tf.placeholder(tf.float32, [None, 784])
	y = tf.placeholder(tf.int64, [None])
	dropout_rate = tf.placeholder(tf.float32)

	train_step, accuracy, learning_rate,_ = cnn_model(x, y, dropout_rate)

	saver = tf.train.Saver()
	with tf.Session() as sess:
		sess.run(tf.global_variables_initializer())
		saver.restore(sess,"./model/model.ckpt")
		for epoch in range(30):
			batch_index = 0
			for _ in range(int(len(train_x) / BATCH_SIZE)):
				batch_index += BATCH_SIZE
				batch_y, batch_x = train_y[batch_index - BATCH_SIZE:batch_index], train_x[batch_index - BATCH_SIZE:batch_index]
				print(batch_index)
				sess.run(train_step, feed_dict={x: batch_x, y: batch_y, dropout_rate: 0.5})
				if not (_+1)%100:
					vb_index = 0
					ac = 0
					for i in range(int(len(verify_x) / BATCH_SIZE)):
						vb_index += BATCH_SIZE
						vby, vbx = verify_y[vb_index - BATCH_SIZE:vb_index], verify_x[vb_index - BATCH_SIZE:vb_index]
						ac += sess.run(accuracy, feed_dict={x: vbx, y: vby, dropout_rate: 0.0})
					lr=sess.run(learning_rate)
					print(epoch, ac / int(len(verify_x) / BATCH_SIZE),lr)
					saver.save(sess, "model/model.ckpt")



def predict(_):
	print("prediction:\n")
	data = pd.read_csv('data/test.csv')
	test_x = data.values
	del data

	test_x = test_x / 255

	x = tf.placeholder(tf.float32, [None, 784])
	y = tf.placeholder(tf.int64, [None])
	dropout_rate = tf.placeholder(tf.float32)

	_,_,_,logits = cnn_model(x, y, dropout_rate)

	saver = tf.train.Saver()
	results=[]
	with tf.Session() as sess:
		sess.run(tf.global_variables_initializer())
		saver.restore(sess,"./model/model.ckpt")
		index = 0
		for _ in range(int(len(test_x) / BATCH_SIZE)):
			index += BATCH_SIZE
			tbx = test_x[index - BATCH_SIZE:index]
			ans= sess.run(logits, feed_dict={x: tbx, y: [0], dropout_rate: 0.0})
			for re in ans:
				results.append(np.argmax(re))

	results = pd.Series(results, name="Label")
	submission = pd.concat([pd.Series(range(1, 28001), name="ImageId"), results], axis=1)

	submission.to_csv("results2.csv", index=False)

if __name__ == "__main__":
	tf.app.run(main=predict)
