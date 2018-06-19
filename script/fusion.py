# -*- coding: UTF-8 -*-
import tensorflow as tf
import string
import numpy as np
import time
import json
import tensorboard as tb
from sklearn.metrics import roc_auc_score
from features import features
import logging
logging.getLogger().setLevel(logging.DEBUG)
logging.basicConfig(
    level=logging.DEBUG,
    format='%(levelname)s[%(asctime)s]%(message)s')

with open('fusion_result.json', 'r') as f:
    d = json.load(f)
x = []
ytrue = []
ideal = 0.0
# for a1 in range(10):
#     print "A1!!!"
#     for b1 in range(10):
#         for c1 in range(10):
#             a = 0.1 * a1
#             b = 0.1 * b1
#             c = 0.1 * c1
#             m = 1 - a - b - c
#             if m < 0:
#                 continue
#             for idx in range(len(d['label'])):
#                 ytrue.append(d['label'][idx])
#                 val = (a * d['svm_st'][idx] + b * d['lstm'][idx] + c * d['xgb_st'][idx] + m * d['xgb_wv'][idx])
#                 x.append(val)
#             curr = roc_auc_score(ytrue, x)
#             if curr > ideal:
#                 ideal = curr
#                 print "%f, %f, %f, %f-------%f" % (a, b, c, m, curr)


class Data():
    def __init__(self):
        with open('fusion_result.json', 'r') as f:
            d = json.load(f)
        self.data = []
        self.batch_id = 0
        self.data_label = d['label']
        for i in range(len(d['label'])):
            item = [d['svm_st'][i], d['lstm'][i], d['xgb_wv'][i], d['xgb_st'][i]]
            self.data.append(item)
            self.batch_id += 1

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

fusion_data = Data()

x_in = tf.placeholder(tf.float32, [None, 4])
y = tf.placeholder(tf.float32, [None])

with tf.variable_scope('fc') as scope:
    weights = {
        'in': tf.Variable(tf.constant(0.25, shape=[4, 1])),
    }
    biases = {
        'in': tf.Variable(tf.constant(0.1, shape=[1, ])),
    }
    scope.reuse_variables()

X = tf.matmul(x_in, weights['in'])
pred = X
# pred = tf.nn.relu(X)
# pred = tf.nn.sigmoid(X)
# cost = -tf.reduce_mean(y * tf.log(tf.nn.relu(pred) + 0.00001) + (1 - y) * tf.log(tf.nn.relu(1 - pred) + 0.00001))
# cost = tf.reduce_sum(tf.cast(tf.equal(pred, y), tf.float32))
cost = tf.reduce_mean(y * tf.nn.relu(0.9 - pred) + (1 - y) * tf.nn.relu(pred))

train_op = tf.train.AdamOptimizer(0.001).minimize(cost)
init_op = tf.global_variables_initializer()
batch_size = 128
iterations = 5000000
display_step = 100

with tf.Session() as sess:
    sess.run(init_op)
    step = 0
    while step * batch_size < iterations:
        batch_x, batch_y = fusion_data.next(batch_size)
        step += 1
        _ = sess.run([train_op], feed_dict={x_in: batch_x, y: batch_y})
        if step % display_step == 0:
            y_pred, y_label, batch_loss = sess.run([pred, y, cost],
                                                   feed_dict={x_in: batch_x, y: batch_y,})
            # print y_pred[:5]
            logging.info("Iter " + str(step * batch_size) + ", Minibatch Loss= " + \
                         "{:.5f}".format(batch_loss) + ", Training Accuracy= " + \
                         "{:.5f}".format(roc_auc_score(y_label, y_pred)))
            w = sess.run(weights)
            print w['in']

