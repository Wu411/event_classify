import pandas as pd
import numpy as np
from get_keyword_new import load_data,getkeyword,keywords_dict
from get_bert import getbert
import tensorflow as tf
import similarity
from clusters_classify import cul_simlarity

import jieba

#打开类别标签维度表
with open('group_label_bert_size.txt', 'r') as read:
    for size in read.readlines():
        size = size.strip('\n')
        size = size.lstrip('(')
        size = size.rstrip(')')
        dim = size.split(', ')
        dim1 = int(dim[0])
        dim2 = int(dim[1])
        dim3 = int(dim[2])
with open('group_threshold.txt','w') as f1:
    group_threshold=[]
    for i in f1.readlines():
        i.strip('\n')
        threshold=float(i)
        group_threshold.append(threshold)
groups_label_bert = np.loadtxt('group_label_bert.txt', delimiter = ',').reshape((dim1, dim2, dim3))

def load_correct_event_classify(path):
    df=pd.read_csv(path)
    new_groups_num=df['cluster_id'].values.tolist()
    old_groups_num=df['old_group_id'].values.tolist()
    for i,j in enumerate(old_groups_num):
        tmp=j.lstrip('[')
        tmp=tmp.rstrip(']')
        tmp=tmp.split(', ')
        old_groups_num[i]=tmp

    summary, fixkeyword = load_data(path)  # 读取并处理数据summary
    # 获取每条数据关键词
    res = []
    for i, j in zip(fixkeyword, summary):
        res.append(getkeyword(i, j, keywords_dict))
        # 提取每条数据关键词词向量
    print('开始提取')
    # 根据提取特征的方法获得词向量
    output = getbert(res)
    print('提取完成')
    return output,new_groups_num,old_groups_num

def events_process(key,event_bert,threshold,flag):
    query = tf.convert_to_tensor(np.asarray(event_bert).reshape(1, 1, 768))
    result = tf.keras.layers.Attention()([query, key])
    with tf.Session() as sess:
        result = result.eval()
    s = similarity.cosSim(event_bert, result[0][0])
    if flag==True:
        if s<threshold:
            threshold=s
        return threshold
    else:
        if s>threshold:
            threshold=s
        return threshold


if __name__=="__name__":
    path = 'D:\\毕设数据\\数据\\train.csv'
    events_bert,new_groups_num_list,old_groups_num_list=load_correct_event_classify(path)
    key_list=tf.convert_to_tensor(groups_label_bert)
    for event,new_group,old_group in zip(events_bert,new_groups_num_list,old_groups_num_list):
        new_group_threshold=events_process(key_list[new_group],event,group_threshold[new_group],True)
        group_threshold[new_group] = new_group_threshold
        for i in old_group:
            if i !=new_group:
                new_group_threshold = events_process(key_list[old_group], event, group_threshold[old_group],False)
                group_threshold[new_group] = new_group_threshold
    with open('group_threshold.txt','w') as f:
        for i in group_threshold:
            f.writelines(str(i)+'\n')
