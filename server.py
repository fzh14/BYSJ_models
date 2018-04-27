# -*- coding:utf-8 -*-

import numpy as np
import xgboost as xgb
from sklearn import svm
from sklearn import linear_model
import numpy as np
import pickle
import jieba
import random
from gensim.models import Word2Vec
import codecs
import sys

from features import features
from sklearn import metrics
from flask import Flask, request, make_response, jsonify, send_from_directory
import traceback
import json


wv = Word2Vec.load('60/Word60.model')
LENGTH = 60
obj = features()
app = Flask(__name__)

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
        s1 = jieba.lcut(li[1])
        s2 = jieba.lcut(li[2])
        msg1 = np.zeros((LENGTH), dtype=np.float32)
        msg2 = np.zeros((LENGTH), dtype=np.float32)
        for i in s1:
            try:
                msg1 += wv[i]
            except:
                pass
        for i in s2:
            try:
                msg2 += wv[i]
            except:
                pass
        msg1_ave = msg1/float(len(s1))
        msg2_ave = msg2/float(len(s2))
        msg = np.concatenate((msg1_ave, msg2_ave))
        x_set.append(msg)
        line_num += 1
    print len(y_label)
    print 'data process finish(word2vec)'
    X = np.array(x_set)
    X = np.reshape(X, (-1, 2*LENGTH))
    y = np.array(y_label)
    return X,y


def test_xgb(q1, q2, name):
    '''
    method = {'xgb_st': load_statistical, 'xgb_wv': load_wv}
    bst = xgb.Booster({'nthread':4}) #init model
    bst.load_model("60/"+name+".model") # load data
    X,y = method[name]('testset.txt')
    dtest=xgb.DMatrix(X)
    ypred=bst.predict(dtest)
    print 'AUC: %.4f' % metrics.roc_auc_score(y, ypred)
    '''
    bst = xgb.Booster({'nthread': 4})  # init model
    bst.load_model("script/60/" + name + ".model")  # load data
    method = {'xgb_st': load_statistical, 'xgb_wv': load_wv}
    if name == 'xgb_st':
        X = []
        for a,b in zip(q1, q2):
            print a,b
            X.append(obj.statistical(a, b))
        #X.append(obj.statistical(q1, q2))
        dtest = xgb.DMatrix(np.array(X))
        ypred = bst.predict(dtest)
        return ypred.tolist()
    elif name == 'xgb_wv':
        X = []
        #for a, b in zip(q1, q2):
        #    X.append(obj.word2vec(a, b))
        X.append(obj.word2vec(q1,q2))
        dtest = xgb.DMatrix(np.array(X))
        ypred = bst.predict(dtest)
        return list(ypred)


def test_svm(q1, q2, name):
    method = {'svm_st': load_statistical, 'svm_wv': load_wv}
    X, y = method[name]('testset.txt')
    with open('fzh1/%s.pickle' % name, 'rb') as f:
        clf = pickle.load(f)
        ypred = clf.predict(X)
        print 'AUC: %.4f' % metrics.roc_auc_score(y, ypred)


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
        for i in query:
            print i
    except:
        traceback.print_exc()
        return make_response(jsonify({'status': 502, 'info': 'format error'}))

    try:
        result = {'status': 200, 'hits': 1}
        result['score'] = test_xgb(query, questions, 'xgb_st')
        return make_response(jsonify(result))

    except BaseException, e:
        traceback.print_exc()
        return make_response(jsonify({'status': 501, 'info': 'system error'}))



if __name__ == '__main__':
    app.run(host = '0.0.0.0', port=6666, threaded=True)
    #train_xgb_st()
    #test_xgb(0, 0, 'xgb_st')
    #test_svm(0, 0, 'svm_st')


