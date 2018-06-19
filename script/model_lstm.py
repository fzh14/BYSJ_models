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


class Data():
    def __init__(self, path):
        obj = features()
        STEPS = 47
        self.data = [] # (total, 2, 47, 60)
        self.batch_id = 0
        self.data_label = []
        self.data_length = []
        file = open(path, 'r')
        num = 0
        for line in file.readlines():
            li = line.strip().split('\t')
            if not len(li) > 0:
                continue
            if li[0] == '1':
                self.data_label.append(1)
            else:
                self.data_label.append(0)
            _t, _seq_len = obj.word2vec_list(li[1], li[2])
            t1 = _t[0]
            t2 = _t[1]
            while len(t1) < STEPS:
                t1.append(np.zeros((60), dtype=np.float32))
            while len(t2) < STEPS:
                t2.append(np.zeros((60), dtype=np.float32))
            self.data.append([t1, t2])
            self.data_length.append(_seq_len)
            num += 1
            # if num == 200:
            #     break


    def next(self, batch_size):
        if self.batch_id + batch_size >= len(self.data):
            batch_data = self.data[self.batch_id: len(self.data)]
            batch_data_label = self.data_label[self.batch_id: len(self.data)]
            batch_seq_len = self.data_length[self.batch_id: len(self.data)]
            self.batch_id = self.batch_id + batch_size - len(self.data)
            batch_data += self.data[0:self.batch_id]
            batch_data_label += self.data_label[0:self.batch_id]
            batch_seq_len += self.data_length[0:self.batch_id]
        else:
            batch_data = self.data[self.batch_id: self.batch_id + batch_size]
            batch_data_label = self.data_label[self.batch_id: self.batch_id + batch_size]
            batch_seq_len = self.data_length[self.batch_id: self.batch_id + batch_size]
            self.batch_id = self.batch_id + batch_size
        return batch_data, batch_data_label, batch_seq_len

class Dssm_w2v():
    def __init__(self):
        self.obj = features()
        step = 1
        learning_rate = 0.001
        self.STEPS = 47
        n_input = 60  # data input
        n_hidden = 64  # hidden layer num of features
        n_classes = 32  # total classes output lstm
        self.X_in = tf.placeholder(tf.float32, [None, 2, self.STEPS, n_input])
        self.y = tf.placeholder(tf.float32, [None])
        self.z = tf.placeholder(tf.int32, [None, 2])
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
        seq1_len = self.z[:, 0]
        seq2_len = self.z[:, 1]
        X1 = tf.reshape(X1, [-1, n_input])
        X1 = tf.matmul(X1, weights['in']) + biases['in']
        X1 = tf.reshape(X1, [-1, self.STEPS, n_hidden])
        X2 = tf.reshape(X2, [-1, n_input])
        X2 = tf.matmul(X2, weights['in']) + biases['in']
        X2 = tf.reshape(X2, [-1, self.STEPS, n_hidden])
        X1 = tf.nn.relu(X1)
        X2 = tf.nn.relu(X2)
        size = tf.shape(seq1_len)[0]
        with tf.name_scope("layer1"):
            with tf.variable_scope("rnn_1"):
                lstm_cell = tf.contrib.rnn.BasicLSTMCell(32)
                init_state = lstm_cell.zero_state(size, dtype=tf.float32)
                outputs1, _1 = tf.nn.dynamic_rnn(lstm_cell, X1, sequence_length=seq1_len, dtype=tf.float32,
                                                 initial_state=init_state)
                outputs2, _2 = tf.nn.dynamic_rnn(lstm_cell, X2, sequence_length=seq2_len, dtype=tf.float32,
                                                 initial_state=init_state)
                outputs1 = tf.nn.relu(outputs1)
                outputs2 = tf.nn.relu(outputs2)

        with tf.name_scope("layer2"):
            with tf.variable_scope("rnn_2"):
                lstm_cell_b = tf.contrib.rnn.BasicLSTMCell(32)
                init_state_b = lstm_cell_b.zero_state(size, dtype=tf.float32)
                __, states1 = tf.nn.dynamic_rnn(lstm_cell_b, outputs1, sequence_length=seq1_len, dtype=tf.float32,
                                                initial_state=None)
                __, states2 = tf.nn.dynamic_rnn(lstm_cell_b, outputs2, sequence_length=seq2_len, dtype=tf.float32,
                                                initial_state=None)

        norm1 = tf.sqrt(tf.reduce_sum(tf.square(states1[1]), axis=1))
        norm2 = tf.sqrt(tf.reduce_sum(tf.square(states2[1]), axis=1))
        dot = tf.reduce_sum(tf.multiply(states1[1], states2[1]), axis=1)
        final = dot / tf.add(tf.multiply(norm1, norm2), biases['mini'])
        with tf.name_scope("prediction"):
            final = final * 0.5 + 0.5
            self.pred = tf.reshape(final, [-1])
        with tf.name_scope("cost"):
            tv = tf.trainable_variables()
            # l2_cost = 0.00001 * tf.reduce_sum([tf.nn.l2_loss(v) for v in tv])
            self.cost = -tf.reduce_mean(self.y * tf.log(self.pred + 0.00001) + (1 - self.y) * tf.log(1 - self.pred + 0.00001))
        self.train_op = tf.train.AdamOptimizer(learning_rate).minimize(self.cost)
        init_op = tf.global_variables_initializer()
        # merged_summary = tf.summary.merge_all()
        self.saver = tf.train.Saver(tf.global_variables())
        # config = tf.ConfigProto()
        # config.gpu_options.per_process_gpu_memory_fraction = 0.6
        self.sess = tf.Session()
        self.sess.run(init_op)
        ckpt = tf.train.get_checkpoint_state('/Users/ivanfzh/Desktop/graduation_proj/fzh/save_w2v/')
        self.saver.restore(self.sess, ckpt.model_checkpoint_path)


    def test(self, q1, q2):
        data = []
        long = []
        for a,b in zip(q1, q2):
            _t, _seq_len = self.obj.word2vec_list(a, b)
            t1 = _t[0]
            t2 = _t[1]
            while len(t1) < self.STEPS:
                t1.append(np.zeros((60), dtype=np.float32))
            while len(t2) < self.STEPS:
                t2.append(np.zeros((60), dtype=np.float32))
            data.append([t1, t2])
            long.append(_seq_len)
        t_pred = self.sess.run(self.pred, feed_dict={self.X_in: data, self.z: long})
        return t_pred


    def train(self, path):
        step = 0
        training_iters = 2000000
        batch_size = 128
        display_step = 100
        trainset = Data('trainset.txt')
        testset = Data('testset.txt')
        print 'load file finish'
        last_acc = 0.0
        while step * batch_size < training_iters:
            batch_x, batch_y, batch_seqlen = trainset.next(batch_size)
            _ = self.sess.run([self.train_op], feed_dict={self.X_in: batch_x, self.y: batch_y, self.z: batch_seqlen})
            # train_writer.add_summary(summary=summary, global_step=step)
            step += 1
            if step % display_step == 0:
                y_pred, y_label, batch_loss = self.sess.run([self.pred, self.y, self.cost],
                                                       feed_dict={self.X_in: batch_x, self.y: batch_y, self.z: batch_seqlen})
                logging.info("Iter " + str(step * batch_size) + ", Minibatch Loss= " + \
                             "{:.5f}".format(batch_loss) + ", Training Accuracy= " + \
                             "{:.5f}".format(roc_auc_score(y_label, y_pred)))
                if step % (display_step * 4) == 0:
                    t_x, t_y, t_seqlen = testset.next(len(testset.data))
                    t_pred = self.sess.run(self.pred, feed_dict={self.X_in: t_x, self.y: t_y, self.z: t_seqlen})
                    acc = roc_auc_score(t_y, t_pred)
                    if acc > last_acc:
                        logging.info("test acc: " + str(acc))
                        print self.saver.save(self.sess, '/Users/ivanfzh/Desktop/graduation_proj/fzh/save_w2v/dssm_test', global_step=step)
                        last_acc = acc
        logging.info('finish training!')

    def fusion(self):
        testset = Data('testset.txt')
        print 'load file finish'
        t_x, t_y, t_seqlen = testset.next(len(testset.data))
        t_pred = self.sess.run(self.pred, feed_dict={self.X_in: t_x, self.y: t_y, self.z: t_seqlen})
        acc = roc_auc_score(t_y, t_pred)
        logging.info("test acc: " + str(acc))
        d = {}
        d['lstm'] = t_pred.tolist()
        with open('fusion_result.json', 'w') as f:
            json.dump(d, f)



if __name__ == "__main__":
    t = Dssm_w2v()
    t.fusion()
