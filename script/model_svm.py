# -*- coding:utf-8 -*-
from sklearn import svm
from sklearn import linear_model
import numpy as np
import pickle
import jieba
import random
from features import features
from sklearn import metrics

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
    print y_label[1]
    print 'word2vec data process finish:'
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


def score_svm(data):
    x_set = []
    for i in data:
        s1 = i['query']
        s2 = i['question']
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
        msg1 = msg1 / float(len(s1))
        msg2 = msg2 / float(len(s2))
        msg = np.concatenate((msg1, msg2))
        x_set.append(msg)
    X = np.array(x_set)
    X = np.reshape(X, (-1, 2*LENGTH))
    with open('clf.pickle', 'rb') as f:
        clf2 = pickle.load(f)
        return clf2.predict(X)


if __name__ == "__main__":
    print 'statistical:'
    train_svm_statistical()
    #print 'w2v'
    #train_svm_wv()

    X_test, y_test = load_statistical('testset.txt')
    with open('save/svm_st.pickle', 'rb') as f:
        clf2 = pickle.load(f)
        ypred = clf2.predict(X_test)
        print 'AUC: %.4f' % metrics.roc_auc_score(y_test, ypred)

    # X_test, y_test = load_w2v('testset.txt')
    # with open('save/svm_wv.pickle', 'rb') as f:
    #     clf2 = pickle.load(f)
    #     ypred = clf2.predict(X_test)
    #     print 'AUC: %.4f' % metrics.roc_auc_score(y_test, ypred)