# -*- coding:utf-8 -*-
from sklearn import svm
from gensim.models.keyedvectors import KeyedVectors
from sklearn import linear_model
import numpy as np
import pickle
import jieba
import random
import codecs
import json
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from scipy.linalg import norm


class features():
    def __init__(self):
        with open('60/word_60.pickle', 'rb') as temp:
            self.wv = pickle.load(temp)
        self.WORD2VEC_LENGTH = 60
        self.d = json.load(codecs.open('share_words.json','r','utf-8'), encoding='utf8')


    def tfidf(self, query1, query2):
        #print 'input:' + query1 + '\t' + query2
        try:
            corpus = [[],[]]
            corpus[0] = ' '.join(jieba.cut(query1))
            corpus[1] = ' '.join(jieba.cut(query2))
            #print corpus
            vectorizer = TfidfVectorizer()
            tfidf_ = vectorizer.fit_transform(corpus)
            vectors = tfidf_.toarray()
            if (norm(vectors[0]) * norm(vectors[1])) != 0:
                return (np.dot(vectors[0], vectors[1])+0.001) / ((norm(vectors[0]) * norm(vectors[1]))+0.001)
            else:
                return 0
        except:
            return 0


    def jaccard_similarity(self, s1, s2):
        """
        计算两个句子的雅可比相似度
        :param s1:
        :param s2:
        :return:
        """
        try:
            corpus = [[], []]
            corpus[0] = ' '.join(jieba.cut(s1))
            corpus[1] = ' '.join(jieba.cut(s2))
            vectorizer = CountVectorizer()
            vectors = vectorizer.fit_transform(corpus).toarray()
            numerator = np.sum(np.min(vectors, axis=0))
            denominator = np.sum(np.max(vectors, axis=0))
            return (1.0 * numerator+0.001) / (denominator+0.001)
        except:
            return 0


    def word2vec(self, query1, query2):
        """
        input 2 query
        return a numpy vector
        """
        s1 = jieba.lcut(query1)
        s2 = jieba.lcut(query2)
        msg1 = np.zeros((self.WORD2VEC_LENGTH), dtype=np.float32)
        msg2 = np.zeros((self.WORD2VEC_LENGTH), dtype=np.float32)
        for i in s1:
            try:
                msg1 += np.array(self.wv[i])
            except:
                pass
        for i in s2:
            # msg2 += self.wv[i]
            try:
                # print i, self.wv[i], len(self.wv[i])
                msg2 += self.wv[i]
            except:
                print i
                pass
        msg1_ave = msg1 / len(s1)
        msg2_ave = msg2 / len(s2)
        msg = np.concatenate((msg1_ave, msg2_ave))
        return msg


    def calculate_sharewords(self, query1, query2):
        ratio = 1.0
        share_word = False
        word_l1 = jieba.lcut(query1)
        word_l2 = jieba.lcut(query2)
        vac1 = {}
        for i in word_l1:
            if not vac1.has_key(i):
                vac1[i] = 0
        for i in word_l2:
            if vac1.has_key(i):
                if self.d.has_key(i):
                    share_word = True
                    ratio = ratio * (1 - self.d[i])
        if share_word:
            return ratio
        else:
            return 0.5


    def show_plt_sharewords(self):
        num = 0
        f1 = open('data.txt', 'r')
        x_total = 40
        x = []
        y0_label = [0 for k in range(x_total)]
        y1_label = [0 for k in range(x_total)]
        for i in range(x_total):
            x.append(i / (x_total * 1.0))
        for line in f1.readlines():
            num += 1
            li = line.split('\t')
            ratio = self.calculate_sharewords(li[1], li[2])
            if ratio >= 1:
                ratio = ratio - 0.0001
            if li[0] == '1':
                y1_label[int(ratio * x_total)] += 1
            else:
                y0_label[int(ratio * x_total)] += 1

        x = np.array(x)
        y0_label = np.array(y0_label)
        y1_label = np.array(y1_label)
        p2 = plt.bar(x, y1_label, 0.025)
        p1 = plt.bar(x, y0_label, 0.025, color='#d62728', bottom=y1_label)
        plt.ylabel('NUM')
        plt.xlabel('Score')
        plt.title('Scores statistic by word share')
        plt.xticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
        plt.legend((p1[0], p2[0]), ('Negative', 'Positive'))
        plt.show()


    def statistical(self, q1, q2):
        '''
        vector ==> [sharewords, jaccard, tfidf_cosine, dtw, len_1/len_2, len1, len2, dtw/(sum)]
        '''
        vector = []
        vector.append(1-self.calculate_sharewords(q1, q2))
        vector.append(self.jaccard_similarity(q1, q2))
        vector.append(self.tfidf(q1, q2))
        vector.append(self.dtw(q1, q2))
        if q1.__class__ == 'a'.__class__:
            q1 = unicode(q1, 'utf-8')
            q2 = unicode(q2, 'utf-8')
        vector.append(1.0*len(q1)/len(q2))
        vector.append(1.0*len(q1))
        vector.append(1.0*len(q2))
        vector.append(vector[-4] / (len(q1)+len(q2)))
        return np.array(vector, dtype=np.float32)

    def dtw(self, q1, q2):
        s1 = jieba.lcut(q1)
        s2 = jieba.lcut(q2)
        msg1 = []
        msg2 = []
        for i in s1:
            try:
                msg1.append(self.wv[i])
            except:
                msg1.append(np.zeros((self.WORD2VEC_LENGTH), dtype=np.float32))
        for i in s2:
            try:
                msg2.append(self.wv[i])
            except:
                print i
                msg2.append(np.zeros((self.WORD2VEC_LENGTH), dtype=np.float32))
        m = len(msg1)
        n = len(msg2)
        matrix = []
        for x in range(m):
            matrix.append([])
            for y in range(n):
                matrix[x].append((np.dot(msg1[x], msg2[y]) + 0.01) / (norm(msg1[x]) * norm(msg2[y]) + 0.01))
        alpha = 0.95
        val = 0
        i, j = 0, 0
        while i < m - 1 and j < n - 1:
            curr = 0
            next_i, next_j = i + 1, j + 1
            for _t in range(j + 1, n):
                if matrix[i + 1][_t] > curr:
                    curr = matrix[i + 1][_t] * alpha ** (_t - (j + 1))
                    next_i, next_j = i + 1, _t
            for _t in range(i + 1, m):
                if matrix[_t][j + 1] > curr:
                    curr = matrix[_t][j + 1] * alpha ** (_t - (i + 1))
                    next_i, next_j = _t, j + 1
            i, j = next_i, next_j
            val += curr
        return val



if __name__ == '__main__':
    print 'start:'
    f = open('testset.txt', 'r')
    line = f.readline()
    line = f.readline()
    li = line.split('\t')
    obj = features()
    # print obj.statistical(li[1], li[2])
    # print obj.dtw(li[1], li[2])
    print obj.word2vec(li[1], li[2])

