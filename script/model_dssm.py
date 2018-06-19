# -*- coding: UTF-8 -*-
import tensorflow as tf
import string
import numpy as np
import time
import sys
import json
from sklearn.metrics import roc_auc_score
import logging
logging.getLogger().setLevel(logging.DEBUG)
logging.basicConfig(
    level=logging.DEBUG,
    format='%(levelname)s[%(asctime)s]%(message)s')
##### prepare data

SLIDING_STEP = 3
STEPS = 20
WORD_DICT = 1149

class Data():
    def __init__(self, path):
        self.data = [] # (total, 2, 21, 1150)
        self.batch_id = 0
        self.data_label = []
        self.data_length = []
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
            r1, l1 = self.input_layer(li[1])
            r2, l2 = self.input_layer(li[2])
            _t.append(r1)
            _t.append(r2)
            self.data.append(_t)
            self.data_length.append([l1, l2])
            ## control mini data
            # num+=1
            # if num == 500:
            #     break

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
        length = len(result)
        while len(result) < STEPS:
            result = [[0 for _ in range(1150)]] + result
        # result += [final]
        return result, length


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

testset = Data('testset.txt')
trainset = Data('trainset.txt')
# print len(trainset.data)
print 'load file finish'

right = 0
total = 0

# ==============
#     MODEL
# ==============

learning_rate = 0.001
training_iters = 5000000
batch_size = 128
display_step = 100

# Network Parameters
n_input = 1150  # data input
n_steps = STEPS+1  # time steps
n_hidden = 32  # hidden layer num of features
n_classes = 32  # total classes output lstm

X_in = tf.placeholder(tf.float32, [None, 2, n_steps, n_input])
y = tf.placeholder(tf.float32, [None])

with tf.variable_scope('fc') as scope:
    weights = {
        # (60 * 300)
        'in': tf.Variable(tf.random_normal([n_input, n_hidden]), ),
    }
    biases = {
        'in': tf.Variable(tf.constant(0.1, shape=[n_hidden, ])),
        'mini': tf.constant(0.00001, shape=[1, ])
    }
    scope.reuse_variables()

def LSTMRNN(X, weights, biases):
    ### x shape (batch, 2, n_steps, n_input)
    X1 = X[:, 0]
    X2 = X[:, 1]
    X1 = tf.reshape(X1, [-1, n_input])
    X1 = tf.matmul(X1, weights['in']) + biases['in']
    X1 = tf.reshape(X1, [-1, n_steps, n_hidden])
    X2 = tf.reshape(X2, [-1, n_input])
    X2 = tf.matmul(X2, weights['in']) + biases['in']
    X2 = tf.reshape(X2, [-1, n_steps, n_hidden])
    X1 = tf.nn.relu(X1)
    X2 = tf.nn.relu(X2)
    X1 = tf.unstack(X1, n_steps, 1)
    X2 = tf.unstack(X2, n_steps, 1)
    size = tf.shape(X)[0]
    with tf.name_scope("layer1"):
        with tf.variable_scope("layer1"):
            lstm_cell = tf.contrib.rnn.BasicLSTMCell(n_hidden)
            init_state1 = lstm_cell.zero_state(size, dtype=tf.float32)
            outputs1, states1 = tf.contrib.rnn.static_rnn(lstm_cell, X1, dtype=tf.float32, scope='cell1'
                                                          , initial_state=init_state1)
            outputs2, states2 = tf.contrib.rnn.static_rnn(lstm_cell, X2, dtype=tf.float32, scope='cell1'
                                                          , initial_state=init_state1)
            # m1 = tf.unstack(tf.nn.relu(outputs1), n_steps, 1)
            m1 = []
            for i in outputs1:
                m1.append(tf.nn.relu(i))
            # m2 = tf.unstack(tf.nn.relu(outputs2), n_steps, 1)
            m2 = []
            for i in outputs2:
                m2.append(tf.nn.relu(i))


    with tf.name_scope("layer2"):
        with tf.variable_scope("layer2"):
            lstm_cell2 = tf.contrib.rnn.BasicLSTMCell(n_classes)
            init_state2 = lstm_cell2.zero_state(size, dtype=tf.float32)
            o1, _ = tf.contrib.rnn.static_rnn(lstm_cell2, m1, dtype=tf.float32, scope='cell2', \
                                              initial_state=init_state2)
            o2, _ = tf.contrib.rnn.static_rnn(lstm_cell2, m2, dtype=tf.float32, scope='cell2', \
                                              initial_state=init_state2)

    # outputs (batch*steps, classes) => (batch, steps, classes)
    #outputs = tf.reshape(outputs, [-1, n_steps, n_classes])
    # final shape => (batch, classes)
    result_1 = o1[-1]
    result_2 = o2[-1]
    # merge = tf.concat([result_1, result_2], 1)
    # final = tf.matmul(merge, weights['out']) + biases['out']
    norm1 = tf.sqrt(tf.reduce_sum(tf.square(result_1), axis=1))
    norm2 = tf.sqrt(tf.reduce_sum(tf.square(result_2), axis=1))
    dot = tf.reduce_sum(tf.multiply(result_1, result_2), axis=1)
    final = dot / tf.add(tf.multiply(norm1, norm2), biases['mini'])
    return final


final = LSTMRNN(X_in, weights, biases)
final = final * 0.5 + 0.5
pred = tf.reshape(final, [-1])
# tv = tf.trainable_variables()
# l2_cost = 0.001 * tf.reduce_sum([ tf.nn.l2_loss(v) for v in tv ])
cost = -tf.reduce_mean( y * tf.log(pred+0.00001) + (1 - y) * tf.log(1 - pred+0.00001))
train_op = tf.train.AdamOptimizer(learning_rate).minimize(cost)
init = tf.global_variables_initializer()
saver = tf.train.Saver()

last_acc = 0.0
config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.6
with tf.Session(config=config) as sess:
# with tf.Session() as sess:
    sess.run(init)
    step = 1
    while step * batch_size < training_iters:
        batch_x, batch_y = trainset.next(batch_size)
        sess.run(train_op, feed_dict={X_in: batch_x, y: batch_y})
        step += 1
        if step % display_step == 0:
            y_pred = sess.run(pred, feed_dict={X_in: batch_x, y: batch_y})
            y_label = sess.run(y, feed_dict={y: batch_y})
            batch_loss = sess.run(cost, feed_dict={X_in: batch_x, y: batch_y})
            # print y_pred, y_label
            logging.info("Iter " + str(step * batch_size) + ", Minibatch Loss= " + \
                  "{:.5f}".format(batch_loss) + ", Training Accuracy= " + \
                  "{:.5f}".format(roc_auc_score(y_label, y_pred)))
            if step % (display_step * 4) == 0:
                t_x, t_y = testset.next(len(testset.data))
                t_pred = sess.run(pred, feed_dict={X_in: t_x, y: t_y})
                t_label = sess.run(y, feed_dict={y: t_y})
                acc = roc_auc_score(t_label, t_pred)
                # print (t_pred, t_label)
                logging.info("test acc: " + str(acc))
                if acc > last_acc:
                    saver.save(sess, 'save_bow/dssm_bow', global_step=step)
                    last_acc = acc

    logging.info('finish training!')
