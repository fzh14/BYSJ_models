# -*- coding:utf-8 -*-
import sys
import random
import jieba
import codecs
import pickle
import json
import numpy as np
import matplotlib.pyplot as plt
import csv
import gensim


def get_data():
    data_list = []
    f = open('database.csv','r')
    fw = open('data.txt','w')
    group_id = -1
    count = 0
    csv_file = csv.reader(f)
    for i in csv_file:
        if i[1] == '标准问法':
            group_id += 1
            data_list.append([i[0]])
        elif i[1] == '相似问法':
            data_list[group_id].append(i[0])
            pass
        else:
            print i, count
            break
        count += 1
    print count
    for idx in range(len(data_list)):
        if idx%100 == 5:
            print idx
        curr_n = len(data_list[idx])
        for i in range(curr_n):
            q1 = data_list[idx][i]
            for _ in range(5):
                k = random.randint(0, curr_n-1)
                if curr_n == 1:
                    break
                while k==i:
                    k = random.randint(0, curr_n - 1)
                q2 = data_list[idx][k]
                fw.write('1\t'+q1+'\t'+q2+'\n')
            for _ in range(5):
                k = random.randint(0, group_id)
                while k==idx:
                    k = random.randint(0, group_id)
                limit = len(data_list[k])
                q2 = data_list[k][random.randint(0,limit-1)]
                fw.write('0\t' + q1 + '\t' + q2 + '\n')


def split_set():
    f_train = open('trainset.txt','w')
    f_test = open('testset.txt','w')
    f1 = open('data.txt','r')
    for line in f1.readlines():
        if(random.random()<0.90):
            f_train.write(line)
        else:
            f_test.write(line)


def split_word():
    dict = {}
    index = 0
    f2 = open('sentence_split.txt','w')
    f = open('sentence.txt','r')
    num = 0
    for line in f.readlines():
        num+=1
        s = []
        line = line.strip()
        for i in jieba.cut(line):
            word = i.encode('utf8')
            if not dict.has_key(word):
                dict[word]=index
                index+=1
            s+=i.encode('utf8')+' '
        f2.write(s.strip()+'\n')
    fj = codecs.open('json.txt', 'w', 'utf-8')
    json.dump(dict, fj, encoding='utf8', ensure_ascii=False)
    print index


def word_share():
    f1 = open('data.txt','r')
    x_total = 40
    x=[]
    for i in range(x_total):
        x.append(i/(x_total*1.0))
    share = {}
    for line in f1.readlines():
        li = line.split('\t')
        word_l1 = jieba.lcut(li[1])
        word_l2 = jieba.lcut(li[2])
        vac1 = {}       
        for i in word_l1:
            if not vac1.has_key(i):
                vac1[i]=0
        for i in word_l2:
            if vac1.has_key(i):
                if not share.has_key(i):
                    share[i]=[0,1]###(positive_num, Total_num)
                else:
                    share[i][1]+=1
                if li[0]=='1':
                    share[i][0]+=1
    d = {}
    for i in share.keys():
        if share[i][1] == 1:
            continue
        d[i]=(share[i][0]*1.0+0.01)/(share[i][1]+0.01)# size(Positive)/size(Total)
        if d[i] == 1.0:
            d[i] = 0.99
    with codecs.open('share_words.json', 'w', 'utf-8') as f:
        json.dump(d, f, ensure_ascii=False, encoding='utf-8')
    '''
    f1 = open('../data.txt','r')
    for line in f1.readlines():
        num+=1
        ratio = 1.0
        share_word = False
        li = line.split('\t')
        word_l1 = jieba.lcut(li[1])
        word_l2 = jieba.lcut(li[2])
        vac1 = {}
        for i in word_l1:
            if not vac1.has_key(i):
                vac1[i]=0
        for i in word_l2:
            if vac1.has_key(i):
                share_word = True
                if d.has_key(i):
                    ratio=ratio*(1-d[i])
        #print str(num)+':'+str(ratio)
        #if share_word==False:
        #    continue
        if ratio>=1:
            ratio=ratio-0.0001
        if li[0]=='1':
            y1_label[int(ratio*x_total)]+=1
        else:
            y0_label[int(ratio*x_total)]+=1

    x = np.array(x)
    y0_label=np.array(y0_label)
    y1_label=np.array(y1_label)
    p2 = plt.bar(x, y1_label, 0.025)
    p1 = plt.bar(x, y0_label, 0.025, color='#d62728', bottom=y1_label)
    plt.ylabel('NUM')
    plt.xlabel('Score')
    plt.title('Scores statistic by word share')
    plt.xticks([0,0.2,0.4,0.6,0.8,1.0])
    plt.legend((p1[0], p2[0]), ('Negative', 'Positive'))
    plt.show()
'''


def train():
    sentences = word2vec.Text8Corpus('sentence_split.txt')
    model = word2vec.Word2Vec(sentences, min_count=3, size=100)
    model.save('model1')


def test():
    dict = json.load(codecs.open('json.txt','r','utf-8'), encoding='utf8')
    v1=[]
    v2=[]
    cal_dist={}
    for i in range(5000):
        v1.append(0)
        v2.append(0)
    model = word2vec.Word2Vec.load('model1')
    f = open('testset.txt','r')
    for line in f.readlines():
        cal_dist={}
        li = line.split('\t')
        word_l1 = jieba.lcut(li[1])
        word_l2 = jieba.lcut(li[2])
        for i in word_l1:
            try:
                for similar_word in model.wv.most_similar(i):
                    if not cal_dist.has_key(similar_word[0]):
                        cal_dist[similar_word[0]] = similar_word[1]*1.0
                    else:
                        if cal_dist[similar_word[0]] < similar_word[1]:
                            cal_dist[similar_word[0]] = similar_word[1]
            except:
                pass
        for i in word_l2:
            try:
                for similar_word in model.wv.most_similar(i):
                    if not cal_dist.has_key(similar_word[0]):
                        cal_dist[similar_word[0]] = similar_word[1]
                    else:
                        cal_dist[similar_word[0]] -= similar_word[1]
            except:
                pass
        ans = 0.0
        for key in cal_dist.keys():
            ans += cal_dist[key]*cal_dist[key]
        print ans


def generate_word_dict():
    f = open('data.txt','r')
    voc = {}
    valid_dict = {}
    arr = []
    for i in f.readlines():
        li = i.strip().split('\t')
        for sentence in li[1:]:
            sentence = unicode(sentence, 'utf-8')
            for w in sentence:
                #print i
                if not voc.has_key(w):
                    voc[w] = 1
                else:
                    voc[w] += 1

    for word in voc.keys():
        arr.append((word, voc[word]))
    print len(arr)
    ar = sorted(arr, key=lambda x: x[1], reverse=True)
    print ar
    for i in range(len(ar)):
        valid_dict[ar[i][0]] = i

    with open('bag_words.json','w') as fw:
        json.dump(valid_dict, fw)
'''
    _test = json.load(open('wordbag.json','r'))
    f = open('data.txt', 'r')
    for i in f.readlines():
        li = i.strip().split('\t')
        for sentence in li[1:]:
            sentence = unicode(sentence, 'utf-8')
            for w in sentence:
                #print i
                if not _test.has_key(w):
                    print w, i.strip()
'''


if __name__ == '__main__':
    #split_set()
    #word_share()
    # with open('60/word_60.pickle', 'rb') as temp:
    #     wv = pickle.load(temp)
    # # print wv.__class__
    # wv.save_word2vec_format('test_01.model.txt', binary=False)
    #
    f = open('test_01.model.txt','r')
    f.readline()
    # f = open('test.txt','r')
    fp = open('w2v.txt', 'w')
    d = {}
    # num = 0
    # wrong = 0
    for line in f.readlines():
        li = line.strip().split(' ')
        if len(li) != 61:
            continue
        list_t = []
        for i in li[1:]:
            list_t.append(float(i))
        arr = np.array(list_t)
        if len(arr) != 60:
            print li[0], list_t
            continue
        d[li[0].decode('utf-8')] = arr
        fp.write(line)

        # d[li[0]] = arr

    # with open('60/60_uni.pickle','wb') as t:
    #     pickle.dump(d,t)

    # for line in f.readlines():
    #     li = line.strip().split(' ')
    #     if len(li) != 61:
    #         continue
    #     num += 1
    #     fp.write(line)
    #
    # print 'total:', num
    # print 'wrong', wrong

    # with open('w2v.txt', 'r') as f:
    #     wv = {}
    #     for line in f.readlines():
    #         li = line.strip().split(' ')
    #         list_t = []
    #         for i in li[1:]:
    #             list_t.append(float(i))
    #         arr = np.array(list_t)
    #         wv[li[0]] = arr
    # with open('60/60.pickle', 'rb') as t:
    #     wv = pickle.load(t)
    # print wv[u'高兴']
    # print wv['如何']