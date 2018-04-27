# -*- coding:utf-8 -*-
import xgboost as xgb
from sklearn import svm
import numpy as np
import pickle
import jieba
import codecs
import sys
from features import features
from sklearn import metrics
import json


LENGTH = 60
obj = features()
STATISTICAL_LENGTH = 8


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
    print 'statistical process finish'
    X = np.array(x_set)
    X = np.reshape(X, (-1, STATISTICAL_LENGTH))
    y = np.array(y_label)
    return X, y


def train_xgb_st():
    X, y = load_statistical('trainset.txt')
    X_test, y_test = load_statistical('testset.txt')
    print X.shape
    print y.shape
    dtrain=xgb.DMatrix(X,label=y)
    dtest=xgb.DMatrix(X_test)
    params={'booster':'gbtree',
        'objective': 'binary:logistic',
        'eval_metric': 'auc',
        'max_depth':7,
        'lambda':2,
        'subsample':0.75,
        'colsample_bytree':0.75,
        'min_child_weight':2,
        'eta': 0.025,
        'seed':0,
        'nthread':4,
         'silent':0}
    watchlist = [(dtrain,'train')]
    bst=xgb.train(params,dtrain,num_boost_round=1000,evals=watchlist)
    ypred=bst.predict(dtest)
    print 'AUC: %.4f' % metrics.roc_auc_score(y_test, ypred)
    bst.save_model('save/xgb_st.model') # 用于存储训练出的模型


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


def train_xgb_wv():
    X, y = load_wv('trainset.txt')
    X_test, y_test = load_wv('testset.txt')
    print X.shape
    print y.shape
    dtrain=xgb.DMatrix(X,label=y)
    dtest=xgb.DMatrix(X_test)
    params={'booster':'gbtree',
        'objective': 'binary:logistic',
        'eval_metric': 'rmse',
        'max_depth':8,
        'lambda':2,
        'subsample':0.75,
        'colsample_bytree':0.75,
        'min_child_weight':3,
        'eta': 0.1,
        'seed':0,
        'nthread':4,
         'silent':0}
    watchlist = [(dtrain,'train')]
    bst=xgb.train(params,dtrain,num_boost_round=2000,evals=watchlist)
    ypred=bst.predict(dtest)
    print 'AUC: %.4f' % metrics.roc_auc_score(y_test, ypred)
    bst.save_model('save/xgb_wv.model') # 用于存储训练出的模型


def test_xgb(q1, q2, name):
    bst = xgb.Booster({'nthread': 4})  # init model
    bst.load_model("save/" + name + ".model")  # load data
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
    elif name == 'sgb_wv':
        X = []
        #for a, b in zip(q1, q2):
        #    X.append(obj.word2vec(a, b))
        X.append(obj.word2vec(q1,q2))
        dtest = xgb.DMatrix(np.array(X))
        ypred = bst.predict(dtest)
        return ypred.tolist()


if __name__ == '__main__':
    train_xgb_wv()
    #test_xgb(0, 0, 'xgb_st')
    #test_svm(0, 0, 'svm_st')

