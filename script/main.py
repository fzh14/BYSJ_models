# -*- coding:utf-8 -*-
import xgboost as xgb
from sklearn import svm
from sklearn import linear_model
import numpy as np
import pickle
import jieba
import random
import codecs
import sys
from features import features
from sklearn import metrics
from flask import Flask, request, make_response, jsonify, send_from_directory
import traceback
import json
from model_lstm import *
from model_bow import *


LENGTH = 60
obj = features()
app = Flask(__name__)
lstm_w2v = Dssm_w2v()

with open('save/svm_wv.pickle', 'rb') as f:
    svm_clf = pickle.load(f)

bst_st = xgb.Booster({'nthread': 4})  # init model
bst_st.load_model("save/xgb_st.model")  # load data

bst_wv = xgb.Booster({'nthread': 4})  # init model
bst_wv.load_model("save/xgb_wv.model")  # load data


def load_statistical(path):
    f1 = open(path, 'r')
    y_label = []
    x_set = []
    line_num = 0
    for line in f1.readlines():
        li = line.strip().split('\t')
        if not len(li) == 3:
            continue
        if li[0] == '0':
            y_label.append(0)
        else:
            y_label.append(1)
        msg = obj.statistical(li[1],li[2])
        x_set.append(msg)
        line_num += 1
    print len(y_label)
    print 'data process finish'
    X = np.array(x_set)
    X = np.reshape(X, (-1, 4))
    y = np.array(y_label)
    return X, y


def load_wv(path):
    f1 = open(path, 'r')
    y_label = []
    x_set = []
    line_num = 0
    for line in f1.readlines():
        li = line.strip().split('\t')
        if li[0] == '0':
            y_label.append(0)
        else:
            y_label.append(1)
        msg = obj.word2vec(li[1], li[2])
        x_set.append(msg)
        line_num += 1
    print len(y_label)
    print 'data process finish(word2vec)'
    X = np.array(x_set)
    X = np.reshape(X, (-1, 2*LENGTH))
    y = np.array(y_label)
    return X,y


def test_xgb(q1, q2, name):
    if name == 'xgb_st':
        bst = bst_st
        X = []
        for a,b in zip(q1, q2):
            X.append(obj.statistical(a, b))
        dtest = xgb.DMatrix(np.array(X))
        ypred = bst.predict(dtest)
        return ypred.tolist()
    elif name == 'xgb_wv':
        bst = bst_wv
        X = []
        for a, b in zip(q1, q2):
            X.append(obj.word2vec(a, b))
        dtest = xgb.DMatrix(np.array(X))
        ypred = bst.predict(dtest)
        return ypred.tolist()


def test_svm(q1, q2, name):
    # with open('save/%s.pickle' % name, 'rb') as f:
    #     clf = pickle.load(f)
    clf = svm_clf
    if name == 'svm_st':
        X = []
        for a,b in zip(q1, q2):
            print a,b
            X.append(obj.statistical(a, b))
        ypred = clf.predict(X)
        return ypred.tolist()
    elif name == 'svm_wv':
        X = []
        for a, b in zip(q1, q2):
            X.append(obj.word2vec(a, b))
        ypred = clf.predict(X)
        return ypred.tolist()


@app.route('/')
def hello_world():
    return 'Hello Flask!'

@app.route('/post/query', methods=['POST'])
def sentence_similarity():
    try:
        data = request.get_data()
        json_re = json.loads(data)
        query = json_re['query']
        questions = json_re['questions']
    except:
        traceback.print_exc()
        return make_response(jsonify({'status': 502, 'info': 'format error'}))

    try:
        result = {'status': 200}
        result['score'] = {}
        result['score']['xgb_st'] = test_xgb(query, questions, 'xgb_st')
        result['score']['xgb_wv'] = test_xgb(query, questions, 'xgb_wv')
        # result['score']['svm_st'] = test_svm(query, questions, 'svm_st')
        result['score']['svm_wv'] = test_svm(query, questions, 'svm_wv')
        result['score']['lstm_wv'] = lstm_w2v.test(query, questions).tolist()
        # result['score']['lstm_bow'] = lstm_bow.test(query, questions).tolist()
        return make_response(jsonify(result))

    except BaseException, e:
        traceback.print_exc()
        return make_response(jsonify({'status': 501, 'info': 'system error'}))



if __name__ == '__main__':
    app.run(host = '0.0.0.0', port=6666, threaded=True)
    #train_xgb_st()
    # test = Dssm_bow()
    # test.train()
    # print test.test(["都支持什么付款方式？"], ["支持的支付方式"])

