# -*- coding:utf-8 -*-
from sklearn import svm
from sklearn import linear_model
import numpy as np
import pickle
import jieba
import random
from features import features
from sklearn import metrics
import json

LENGTH = 60
DATA_PATH='data.txt'
obj = features()
STATISTICAL_LENGTH = 8



def split(sent):
    return list(jieba.cut(sent))


def load_w2v(path):
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
    print 'word2vec data process finish'
    X = np.array(x_set)
    X = np.reshape(X, (-1, 2*LENGTH))
    y = np.array(y_label)
    return X,y

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
    print 'statistical process finish!'
    X = np.array(x_set)
    X = np.reshape(X, (-1, STATISTICAL_LENGTH))
    y = np.array(y_label)
    return X, y


def train_svm_wv():
    clf = svm.SVR(verbose=True, shrinking=False)
    X,y = load_w2v('trainset.txt')
    X_test,y_test = load_w2v('testset.txt')
    print X.shape
    print y.shape
    clf.fit(X, y)
    with open('save/svm_wv.pickle', 'wb') as f:
        pickle.dump(clf, f)
    with open('save/svm_wv.pickle', 'rb') as f:
        clf2 = pickle.load(f)
        print clf2.score(X_test, y_test)
    print "mission complete!"


def train_svm_statistical():
    clf = svm.SVR(verbose=True, shrinking=False)
    X, y = load_statistical('trainset.txt')
    X_test, y_test = load_statistical('testset.txt')
    print X.shape
    print y.shape
    clf.fit(X, y)
    with open('save/svm_st.pickle', 'wb') as f:
        pickle.dump(clf, f)
    with open('save/svm_st.pickle', 'rb') as f:
        clf2 = pickle.load(f)
        ypred = clf2.predict(X_test)
        print 'AUC: %.4f' % metrics.roc_auc_score(y_test, ypred)
    print "mission complete!"



def train_lr():
    print "logistic regression:"
    clf = linear_model.LogisticRegression()
    X, y = load_w2v('trainset.txt')
    X_test, y_test = load_w2v('testset.txt')
    print X.shape
    print y.shape
    clf.fit(X, y)
    with open('clf_lr.pickle', 'wb') as f:
        pickle.dump(clf, f)
    with open('clf_lr.pickle', 'rb') as f:
        clf2 = pickle.load(f)
        ypred = clf2.predict(X_test)
        print 'AUC: %.4f' % metrics.roc_auc_score(y_test, ypred)
    print "mission complete!"


if __name__ == "__main__":
    X_test, y_test = load_w2v('testset.txt')
    with open('save/svm_wv.pickle', 'rb') as f:
        clf2 = pickle.load(f)
        ypred = clf2.predict(X_test)
        print 'AUC: %.4f' % metrics.roc_auc_score(y_test, ypred)

    with open('fusion_result.json', 'r') as f:
        d = json.load(f)

    d['svm_st'] = ypred.tolist()
    with open('fusion_result.json', 'w') as f:
        json.dump(d, f)


    # X_test, y_test = load_w2v('testset.txt')
    # with open('save/svm_wv.pickle', 'rb') as f:
    #     clf2 = pickle.load(f)
    #     ypred = clf2.predict(X_test)
    #     print 'AUC: %.4f' % metrics.roc_auc_score(y_test, ypred)