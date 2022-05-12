from collections import defaultdict
import math
import operator
import jieba
import pandas as pd
import re
from bert_serving.client import BertClient
import numpy as np


config_path = 'chinese_L-12_H-768_A-12/bert_config.json'
checkpoint_path = 'chinese_L-12_H-768_A-12/bert_model.ckpt'
dict_path = 'chinese_L-12_H-768_A-12/vocab.txt'
#打开自定义词典
with open("self_dict.txt", "r", encoding='utf-8') as f2:
    self_dict = []
    for line in f2.readlines():
        word = line.strip('\n')
        self_dict.append(word)

def update_selfdict(txt_list,res):#将下划线和连字符所连固定搭配动态加入自定义词典
    for txt in txt_list:
        spec_words = re.findall(r'([a-zA-Z][a-zA-Z0-9]+([-_][a-zA-Z0-9]+)+)', txt)
        for i in spec_words:
            if i[0] not in res:
                res.append(i[0])
    with open("self_dict.txt", "w", encoding='utf-8') as f1:
        for i in res:
            f1.writelines(i)
            f1.write('\n')
    return res

def loadDataSet():
    df = pd.read_excel('D:\\毕设数据\\数据\\副本train3_增加groupname.xlsx', sheet_name="工作表 1 - train")
    tmp_list = df['group_name'].dropna().drop_duplicates().values.tolist()
    update_selfdict(tmp_list,self_dict)
    jieba.load_userdict("self_dict.txt")
    dataset = []
    test=re.compile(r'\W+')
    for w in tmp_list:
        l=[]
        for j in jieba.lcut(w):
            if not test.match(j):
                l.append(j.casefold())
        dataset.append(l)
    return dataset


"""
函数说明：特征选择TF-IDF算法
Parameters:
     list_words:词列表
Returns:
     dict_feature_select:特征选择词字典
"""
#dataset:文件夹，word_list:某一个文件，word某个词

def feature_select(dataset):
    # 总词频统计
    doc_frequency = defaultdict(int) #记录每个词出现的次数，可以把它理解成一个可变长度的list，只要你索引它，它就自动扩列
    for file in dataset:
        for word in file:
            doc_frequency[word] += 1
    '''# 计算每个词的TF值
    word_tf = {}  # 存储没个词的tf值
    for i in doc_frequency:
        word_tf[i] = doc_frequency[i] / sum(doc_frequency.values()) #sum(doc.frequency.values)'''

    # 计算每个词的IDF值
    doc_num = len(dataset)
    word_idf = {}  # 存储每个词的idf值
    word_doc = defaultdict(int)  # 存储包含该词的文档数
    for word in doc_frequency:
        for file in dataset:
            if word in file:
                word_doc[word] += 1
    #word_doc和doc_frequency的区别是word_doc存储的是包含这个词的文档数，即如果一个文档里有重复出现一个词则word_doc < doc_frequency
    for word in doc_frequency:
        word_idf[word] = math.log(doc_num / (word_doc[word] + 1))

    # 计算每个词的TF*IDF的值
    word_tf_idf = {}
    for word in doc_frequency:
        word_tf_idf[word] = word_idf[word] #* word_tf[word]

    # 对字典按值由大到小排序
    dict_feature_select = sorted(word_tf_idf.items(), key=operator.itemgetter(1), reverse=True)
    return dict_feature_select


if __name__ == '__main__':
    data_list = loadDataSet()  # 加载数据
    features = feature_select(data_list)  # 所有词的TF-IDF值
    with open("new_all_tfidf_dict.txt", "w", encoding='utf-8') as f:
        for i in features:
            f.writelines(i[0]+' '+str(i[1]))
            f.write('\n')
    words=[]
    for i in features:
        words.append(i[0])
    bc = BertClient()
    vector = bc.encode(words)

    np.savetxt('keyword_vector_dict.txt',vector)