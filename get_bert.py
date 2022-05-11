#! -*- coding: utf-8 -*-
# 测试代码可用性: 提取特征
import time
from bert_serving.client import BertClient
import pandas as pd
import numpy as np
import get_keyword_new
import os

config_path = 'chinese_L-12_H-768_A-12/bert_config.json'
checkpoint_path = 'chinese_L-12_H-768_A-12/bert_model.ckpt'
dict_path = 'chinese_L-12_H-768_A-12/vocab.txt'

#打开关键词词典
with open("new_all_tfidf_dict.txt", "r", encoding='utf-8') as f:
    keywords_dict={}
    for line in f.readlines():
        line=line.strip('\n')
        #line = line.lstrip('(')
        #line = line.rstrip(')')
        x=line.split(' ')
        keywords_dict[x[0]]=x[1]



#打开自定义词典
with open("self_dict.txt", "r", encoding='utf-8') as f1:
    self_dict = []
    for line in f1.readlines():
        word = line.strip('\n')
        self_dict.append(word)

#打开主机名文件
with open("host_name.txt", "r", encoding='utf-8') as f2:
    host_name = []
    for line in f2.readlines():
        word = line.strip('\n')
        host_name.append(word)

def getbert(data,weight):
    num=0
    output=[]
    start = time.clock()
    for seg_list,wei_list in zip(data,weight):
        num += 1
        if seg_list:
            bc = BertClient()
            new = []
            vector = bc.encode(seg_list)
            for i in range(768):
                temp = 0
                for j in range(len(vector)):
                    temp += vector[j][i]*wei_list[j]
                new.append(temp / (len(vector)))
        else:
            new=[0 for i in range(768)]
        output.append(new)
        print('完成提取：', num)
    end = time.clock()
    print('Running time: %s Seconds' % (end - start))
    return output

if __name__ == "__main__":
    #本程序用于获取现有数据的词向量

    # 读取处理数据
    path='D:\\毕设数据\\数据\\副本train3_增加groupname.xlsx'
    print('数据预处理')
    '''summary, data = get_keyword_new.load_data(path,host_name)  # 读取并处理数据summary

    print('数据预处理完成')
    print("开始获取数据关键词")
    # 获取每条数据关键词
    res_words = []
    res_weights = []
    seg_lists=[]
    for j in summary:
        words,weights=get_keyword_new.getkeyword(j, keywords_dict)
        res_words.append(words)
        res_weights.append(weights)
    print("数据关键词获取结束")
    df = pd.read_excel(path, sheet_name="工作表 1 - train")
    df['keyword_new'] = res_words
    df.to_excel(path, sheet_name="工作表 1 - train")'''
    df = pd.read_excel(path, sheet_name="工作表 1 - train")
    events_keywords = df['keyword_new'].values.tolist()
    keys = df['keyword_new'].drop_duplicates().values.tolist()
    res_words = []
    res_weights = []
    for i in keys:
        i=i.lstrip('[')
        i=i.rstrip(']')
        words=i.split(', ')
        weights=[]
        for j in words:
            weights.append(float(keywords_dict[j.strip('\'')]))
        res_words.append(words)
        res_weights.append(weights)
    print('开始提取')
    output = getbert(res_words, res_weights)
    feature=[]
    emebdding_dict=dict(keys,output)
    for event in events_keywords:
        feature.append(emebdding_dict[event])

    #提取每条数据关键词词向量
    #data=load_data(path)
    # 根据提取特征的方法获得词向量
    print('保存数据')
    df['word_embedding']=feature
    df.to_excel(path,sheet_name="工作表 1 - train")
    #np.savetxt("text_vectors_new1.txt",feature)
    #os.system('pre_cluster_texts.py')