# -*- coding: UTF-8 -*-

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.contrib import rnn

lr = 1e-4
INPUT_SIZE = 28
TIME_SIZE = 28
HIDDEN_UNIT = 256
BATCH_SIZE=100
LAYER_NUM = 2

x_ = tf.placeholder(tf.float32, [None, 784])
y = tf.placeholder(tf.int64, [None])
keep_prob = tf.placeholder(tf.float32)
x = tf.reshape(x_, [-1, 28, 28])

lstm_basic = rnn.BasicLSTMCell(num_units=HIDDEN_UNIT, state_is_tuple=True)
lstm_cell = rnn.DropoutWrapper(cell=lstm_basic, input_keep_prob=1.0, output_keep_prob=keep_prob)
# mlstm_cell = rnn.MultiRNNCell([lstm_cell] * LAYER_NUM, state_is_tuple=True)

init = lstm_cell.zero_state(batch_size=BATCH_SIZE, dtype=tf.float32)

outputs, state = tf.nn.dynamic_rnn(lstm_cell, inputs=x, initial_state=init, time_major=False)

h_state = outputs[:, -1, :]

weights = tf.Variable(tf.truncated_normal([HIDDEN_UNIT, 10], stddev=0.1, dtype=tf.float32))
bias = tf.Variable(tf.constant(0.1, shape=[10], dtype=tf.float32))
logits = tf.matmul(h_state, weights) + bias

cross_entropy = tf.losses.sparse_softmax_cross_entropy(labels=y, logits=logits)
loss = tf.reduce_mean(cross_entropy)  # + LOSS_REGULARIZER(weights)

optimizer = tf.train.AdamOptimizer(lr)
train_step = optimizer.minimize(loss)

correct_prediction = tf.cast(tf.equal(tf.argmax(logits, 1), y), tf.float64)
accuracy = tf.reduce_mean(correct_prediction)
def train():
	print("train:\n")
	data = pd.read_csv('data/train.csv').sample(frac=1)
	train_data = data[0:int(0.8 * len(data))]
	verify_data = data[int(0.8 * len(data)):-1]
	train_y = train_data['label'].values
	train_x = train_data.drop(labels=['label'], axis=1).values
	verify_y = verify_data['label'].values
	verify_x = verify_data.drop(labels=['label'], axis=1).values
	del data, train_data, verify_data,

	train_x = train_x / 255
	verify_x = verify_x / 255

	saver = tf.train.Saver()
	with tf.Session() as sess:
		sess.run(tf.global_variables_initializer())
		saver.restore(sess,"./lstm_model/model.ckpt")
		for epoch in range(30):
			batch_index = 0
			for _ in range(int(len(train_x) / BATCH_SIZE)):
				batch_index += BATCH_SIZE
				batch_y, batch_x = train_y[batch_index - BATCH_SIZE:batch_index], train_x[
				                                                                  batch_index - BATCH_SIZE:batch_index]
				print(batch_index)
				sess.run(train_step, feed_dict={x_: batch_x, y: batch_y, keep_prob: 0.5})
				if not (_ + 1) % 100:
					vb_index = 0
					ac = 0
					for i in range(int(len(verify_x) / BATCH_SIZE)):
						vb_index += BATCH_SIZE
						vby, vbx = verify_y[vb_index - BATCH_SIZE:vb_index], verify_x[vb_index - BATCH_SIZE:vb_index]
						ac += sess.run(accuracy, feed_dict={x_: batch_x, y: batch_y, keep_prob: 1.0})
					print(epoch, ac / int(len(verify_x) / BATCH_SIZE))
					saver.save(sess, "lstm_model/model.ckpt")

def pre():
	print("prediction:\n")
	data = pd.read_csv('data/test.csv')
	test_x = data.values
	del data

	test_x = test_x / 255

	saver = tf.train.Saver()
	results=[]
	with tf.Session() as sess:
		sess.run(tf.global_variables_initializer())
		saver.restore(sess,"./lstm_model/model.ckpt")
		index = 0
		for _ in range(int(len(test_x) / BATCH_SIZE)):
			index += BATCH_SIZE
			tbx = test_x[index - BATCH_SIZE:index]
			ans= sess.run(logits, feed_dict={x_: tbx, y: [0], keep_prob: 1.0})
			for re in ans:
				results.append(np.argmax(re))
			print(results)

	results = pd.Series(results, name="Label")
	submission = pd.concat([pd.Series(range(1, 28001), name="ImageId"), results], axis=1)

	submission.to_csv("results_lstm.csv", index=False)

if __name__ == '__main__':
	pre()
