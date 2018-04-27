# -*- coding: UTF-8 -*-
import tensorflow as tf
import string
import numpy as np
import time
import sys
import json
from sklearn.metrics import roc_auc_score

##### prepare data

SLIDING_STEP = 3
STEPS = 20
WORD_DICT = 1149

class Data():
    def __init__(self, path):
        self.data = [] # (total, 2, 21, 1150)
        self.batch_id = 0
        self.data_label = []
        self.bag_words = json.load(open('bag_words.json', 'r'))
        file = open(path, 'r')
        num = 0
        for line in file.readlines():
            li = line.strip().split('\t')
            if li[0] == '1':
                self.data_label.append(1)
            else:
                self.data_label.append(0)
            _t = []
            _t.append(self.input_layer(li[1]))
            _t.append(self.input_layer(li[2]))
            self.data.append(_t)
            ## control mini data
            num+=1
            if num == 500:
                break

    def input_layer(self, query):
        """
        :param query: str
        :return: a [Steps+1, 1150] array
        """
        sentence = unicode(query, 'utf-8')
        q_label = []
        result = []
        for w in sentence:
            if not self.bag_words.has_key(w):
                print 'No word record!'
                q_label.append(1149)
            else:
                q_label.append(self.bag_words[w])
        final = [0 for k in range(1150)]
        for idx in q_label:
            final[idx] += 1
        for i in range(STEPS):
            if not i < len(q_label) - 2:
                break
            curr = [0 for k in range(1150)]
            for _i in range(3):
                _v = q_label[i + _i]
                curr[_v] += 1
            result.append(curr)
        while len(result) < STEPS:
            result = [[0 for _ in range(1150)]] + result
        result += [final]
        return result


    def next(self, batch_size):
        if self.batch_id + batch_size >= len(self.data):
            batch_data = self.data[self.batch_id: len(self.data)]
            batch_data_label = self.data_label[self.batch_id: len(self.data)]
            self.batch_id = self.batch_id + batch_size - len(self.data)
            batch_data += self.data[0:self.batch_id]
            batch_data_label += self.data_label[0:self.batch_id]
        else:
            batch_data = self.data[self.batch_id: self.batch_id + batch_size]
            batch_data_label = self.data_label[self.batch_id: self.batch_id + batch_size]
            self.batch_id = self.batch_id + batch_size
        return batch_data, batch_data_label


trainset = Data('trainset.txt')
testset = Data('testset.txt')
print len(trainset.data)
print 'load file finish'

right = 0
total = 0

# ==============
#     MODEL
# ==============

learning_rate = 0.001
training_iters = 10000
batch_size = 128
display_step = 10

# Network Parameters
n_input = 1150  # data input
n_steps = STEPS+1  # time steps
n_hidden = 300  # hidden layer num of features
n_classes = 128  # total classes output lstm

X_in = tf.placeholder(tf.float32, [None, 2, n_steps, n_input])
y = tf.placeholder(tf.float32, [None])


# define weights
weights = {
    # (1150 * 300)
    # 'in': tf.Variable(tf.random_normal([n_input, n_hidden]), ),
    # (128*2, 1)
    'out': tf.Variable(tf.random_normal([n_classes * 2, 1]))
}
biases = {
    # 'in': tf.Variable(tf.constant(0.1, shape=[n_hidden, ])),
    'out': tf.Variable(tf.constant(0.1, shape=[1, ]))
}

def LSTMRNN(X, weights, biases):
    ### x shape (batch, 2, steps, n_input)
    X1 = X[:, 0]
    X2 = X[:, 1]

    # X1 = tf.reshape(X1, [-1, n_input])
    # X1 = tf.matmul(X1, weights['in']) + biases['in']
    # X1 = tf.reshape(X1, [-1, n_steps, n_hidden])
    X1 = tf.tanh(X1)
    X1 = tf.unstack(X1, n_steps, 1)

    # X2 = tf.reshape(X2, [-1, n_input])
    # X2 = tf.matmul(X2, weights['in']) + biases['in']
    # X2 = tf.reshape(X2, [-1, n_steps, n_hidden])
    X2 = tf.tanh(X2)
    X2 = tf.unstack(X2, n_steps, 1)

    lstm_cell = tf.contrib.rnn.BasicLSTMCell(n_classes)

    outputs1, states1 = tf.contrib.rnn.static_rnn(lstm_cell, X1, dtype=tf.float32, scope='cell')
    outputs2, states2 = tf.contrib.rnn.static_rnn(lstm_cell, X2, dtype=tf.float32, scope='cell')

    # outputs (batch*steps, classes) => (batch, steps, classes)
    #outputs = tf.reshape(outputs, [-1, n_steps, n_classes])
    # final shape => (batch, classes)
    result_1 = outputs1[-1]
    result_2 = outputs2[-1]
    merge = tf.concat([result_1, result_2], 1)
    final = tf.matmul(merge, weights['out']) + biases['out']
    return final


final = LSTMRNN(X_in, weights, biases)
pred = tf.sigmoid(final)
pred = tf.reshape(pred, [batch_size])
tv = tf.trainable_variables()
l2_cost = 0.001 * tf.reduce_sum([ tf.nn.l2_loss(v) for v in tv ])
cost = -tf.reduce_mean( y * tf.log(pred+0.00001) + (1 - y) * tf.log(1 - pred+0.00001)) + l2_cost
train_op = tf.train.AdamOptimizer(learning_rate).minimize(cost)
init = tf.global_variables_initializer()
saver = tf.train.Saver()

# config = tf.ConfigProto()
# config.gpu_options.per_process_gpu_memory_fraction = 0.4
# with tf.Session(config=config) as sess:
with tf.Session() as sess:
    sess.run(init)
    step = 1
    while step * batch_size < training_iters:
        batch_x, batch_y = trainset.next(batch_size)
        sess.run(train_op, feed_dict={X_in: batch_x, y: batch_y})
        step += 1
        if step % display_step == 0:
            print 'show message:'
            y_pred = sess.run(pred, feed_dict={X_in: batch_x, y: batch_y})
            y_label = sess.run(y, feed_dict={y: batch_y})
            batch_loss = sess.run(cost, feed_dict={X_in: batch_x, y: batch_y})
            print y_pred, y_label
            print "Iter " + str(step * batch_size) + ", Minibatch Loss= " + \
                  "{:.5f}".format(batch_loss) + ", Training Accuracy= " + \
                  "{:.5f}".format(roc_auc_score(y_label, y_pred))


    print 'finish training!'
    saver.save(sess, 'save/dssm_test')

    test_step = 0
    test_acc = 0
    while test_step * batch_size < len(testset.data):
        batch_x, batch_y = testset.next(batch_size)
        y_pred = sess.run(pred, feed_dict={X_in: batch_x, y: batch_y})
        y_label = sess.run(y, feed_dict={y: batch_y})
        acc = roc_auc_score(y_label, y_pred)
        test_acc += acc
        test_step += 1
    print 'test acc:' + str(test_acc / test_step)

