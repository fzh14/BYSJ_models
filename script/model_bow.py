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

class Data():
    def __init__(self, path=""):
        self.STEPS = 20
        self.SLIDING_STEP = 3
        self.WORD_DICT = 1149
        self.data = [] # (total, 2, 21, 1150)
        self.batch_id = 0
        self.data_label = []
        self.data_length = []
        self.bag_words = json.load(open('bag_words.json', 'r'))
        if path == "":
            return
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
            num+=1
            # if num == 200:
            #     break

    def input_layer(self, query):
        """
        :param query: str
        :return: a [Steps+1, 1150] array
        """
        try:
            sentence = unicode(query, 'utf-8')
        except:
            sentence = query
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
        for i in range(self.STEPS):
            if not i < len(q_label) - 2:
                break
            curr = [0 for k in range(1150)]
            for _i in range(3):
                _v = q_label[i + _i]
                curr[_v] += 1
            result.append(curr)
        length = len(result)
        while len(result) < self.STEPS:
            result = [[0 for _ in range(1150)]] + result
        result += [final]
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


class Dssm_bow():
    def __init__(self):
        learning_rate = 0.001
        # Network Parameters
        n_input = 1150  # data input
        n_steps = 21  # time steps
        n_hidden = 32  # hidden layer num of features

        self.X_in = tf.placeholder(tf.float32, [None, 2, n_steps, n_input])
        self.y = tf.placeholder(tf.float32, [None])

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
        X1 = self.X_in[:, 0]
        X2 = self.X_in[:, 1]
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
        size = tf.shape(self.X_in)[0]
        with tf.name_scope("layer1"):
            with tf.variable_scope("layer1"):
                lstm_cell = tf.contrib.rnn.BasicLSTMCell(32)
                init_state1 = lstm_cell.zero_state(size, dtype=tf.float32)
                outputs1, states1 = tf.contrib.rnn.static_rnn(lstm_cell, X1, dtype=tf.float32, scope='cell1'
                                                              , initial_state=init_state1)
                outputs2, states2 = tf.contrib.rnn.static_rnn(lstm_cell, X2, dtype=tf.float32, scope='cell1'
                                                              , initial_state=init_state1)
                m1 = []
                for i in outputs1:
                    m1.append(tf.nn.relu(i))
                m2 = []
                for i in outputs2:
                    m2.append(tf.nn.relu(i))

        with tf.name_scope("layer2"):
            with tf.variable_scope("layer2"):
                lstm_cell2 = tf.contrib.rnn.BasicLSTMCell(32)
                init_state2 = lstm_cell2.zero_state(size, dtype=tf.float32)
                o1, _ = tf.contrib.rnn.static_rnn(lstm_cell2, m1, dtype=tf.float32, scope='cell2', \
                                                  initial_state=init_state2)
                o2, _ = tf.contrib.rnn.static_rnn(lstm_cell2, m2, dtype=tf.float32, scope='cell2', \
                                                  initial_state=init_state2)

        result_1 = o1[-1]
        result_2 = o2[-1]
        norm1 = tf.sqrt(tf.reduce_sum(tf.square(result_1), axis=1))
        norm2 = tf.sqrt(tf.reduce_sum(tf.square(result_2), axis=1))
        dot = tf.reduce_sum(tf.multiply(result_1, result_2), axis=1)
        final = dot / tf.add(tf.multiply(norm1, norm2), biases['mini'])
        final = final * 0.5 + 0.5
        self.pred = tf.reshape(final, [-1])
        # self.cost = tf.reduce_sum(self.y * tf.nn.relu(0.9 - self.pred) + (1 - self.y) * tf.nn.relu(self.pred - 0.1))
        self.cost = -tf.reduce_mean(self.y * tf.log(self.pred + 0.00001) + (1 - self.y) * tf.log(1 - self.pred + 0.00001))
        self.train_op = tf.train.AdamOptimizer(learning_rate).minimize(self.cost)
        init = tf.global_variables_initializer()
        self.saver = tf.train.Saver()
        self.sess = tf.Session()
        self.sess.run(init)
        # ckpt = tf.train.get_checkpoint_state('/Users/ivanfzh/Desktop/graduation_proj/fzh/save_bow/')
        # self.saver.restore(self.sess, ckpt.model_checkpoint_path)

    def train(self):
        training_iters = 3000000
        batch_size = 128
        display_step = 100
        step = 1
        testset = Data('testset.txt')
        print 'load test file finish'
        trainset = Data('trainset.txt')
        print 'load train file finish'
        last_acc = 0.0
        while step * batch_size < training_iters:
            batch_x, batch_y = trainset.next(batch_size)
            self.sess.run(self.train_op, feed_dict={self.X_in: batch_x, self.y: batch_y})
            step += 1
            if step % display_step == 0:
                y_pred, batch_loss = self.sess.run([self.pred, self.cost], feed_dict={self.X_in: batch_x, self.y: batch_y})
                y_label = batch_y
                logging.info("Iter " + str(step * batch_size) + ", Minibatch Loss= " + \
                             "{:.5f}".format(batch_loss) + ", Training Accuracy= " + \
                             "{:.5f}".format(roc_auc_score(y_label, y_pred)))
                if step % (display_step * 4) == 0:
                    t_x, t_y = testset.next(len(testset.data))
                    t_pred = self.sess.run(self.pred, feed_dict={self.X_in: t_x, self.y: t_y})
                    t_label = t_y
                    acc = roc_auc_score(t_label, t_pred)
                    logging.info("test acc: " + str(acc))
                    if acc > last_acc:
                        self.saver.save(self.sess, '/Users/ivanfzh/Desktop/graduation_proj/fzh/save_bow/dssm_test', global_step=step)
                        last_acc = acc
        logging.info('finish training!')


    def test(self, q1, q2):
        data = []
        obj = Data()
        for a, b in zip(q1, q2):
            t1, _ = obj.input_layer(a)
            t2, _ = obj.input_layer(b)
            data.append([t1, t2])
        t_pred = self.sess.run(self.pred, feed_dict={self.X_in: data,})
        return t_pred