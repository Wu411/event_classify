import get_keyword_new
import get_bert
import pandas as pd
import similarity
import numpy as np
import tensorflow as tf
import get_grouplabel_bert
from cluster_texts import cul_simlarity
import os
import schedule
import time

#打开聚类阈值表
with open("clusters_threshold.txt", "r", encoding='utf-8') as f1:
    clusters_threshold = {}
    for line in f1.readlines():
        word = line.strip('\n')
        word=word.split(':')
        key=int(word[0])
        value=float(word[1])
        clusters_threshold[key]=value

#打开聚类类别对照表
with open("clusters_group.txt", "r", encoding='utf-8') as f2:
    clusters_group = {}
    for line in f2.readlines():
        word = line.strip('\n')
        word=word.split(':')
        key=int(word[0])
        tmp = word[1].lstrip('[')
        tmp = tmp.rstrip(']')
        value=tmp.split(', ')
        clusters_group[key]=list(map(int,value))

#打开聚类中心向量表
with open("clusters_center.txt", "r", encoding='utf-8') as f3:
    clusters_center = {}
    for line in f3.readlines():
        word = line.strip('\n')
        word=word.split(':')
        key=int(word[0])
        tmp = word[1].lstrip('[')
        tmp = tmp.rstrip(']')
        value=tmp.split(', ')
        clusters_center[key]=list(map(float,value))
#打开类别标签维度表
with open('group_label_bert_size.txt', 'r') as read:
    for size in read.readlines():
        size = size.strip('\n')
        size = size.lstrip('(')
        size = size.rstrip(')')
        dim = size.split(', ')
        dim1 = dim[0]
        dim2 = dim[1]
        dim3 = dim[2]
with open("noise_num.txt", "r", encoding='utf-8') as f4:
    tmp = f4.read()
    tmp = tmp.split(' ')
    all_num = int(tmp[0])
    noise_num = int(tmp[1])
    noise_per = float(tmp[2])
#打开类别标签向量表
groups_label_bert = np.loadtxt('group_label_bert.txt', delimiter = ',').reshape((dim1, dim2, dim3))

def new_event_getbert(path):
    summary, fixkeyword = get_keyword_new.load_data(path)  # 读取并处理数据summary
    # 获取每条数据关键词
    res = []
    for i, j in zip(fixkeyword, summary):
        res.append(get_keyword_new.getkeyword(i, j, get_keyword_new.keywords_dict))
    output = get_bert.getbert(res)
    return output

def noise_process(noise_point,group_label_dict):
    key_list = tf.convert_to_tensor(groups_label_bert)
    query = tf.convert_to_tensor(np.asarray(noise_point).reshape(1, 1, 768))
    result = tf.keras.layers.Attention()([query, key_list])
    with tf.Session() as sess:
        result = result.eval()
    groups_result, simi = cul_simlarity(noise_point, group_label_dict, result)
    return groups_result,simi

def event_classify(event_bert,group_label_dict,noise_num):
    res=[]
    for new in event_bert:
        flag=False
        tmp=[]
        for label,center in clusters_center.items():
            s = similarity.cosSim(new, center)
            if s < clusters_threshold[label]:
                continue
            else:
                flag=True
                for i in clusters_group[label]:
                    if i not in tmp:
                        tmp.append(i)
        if flag==True:
            res.append(tmp)
        else:
            noise_num+=1
            group_num, simi = noise_process(new, group_label_dict)
            res.append(group_num)
    return res

def event_solution(event_group_num):
    path = 'D:\\毕设数据\\数据\\event_solution.xls'
    df = pd.read_excel(path, sheet_name="Sheet1")
    solu=[]
    for event in event_group_num:
        tmp=[]
        for group_num in event:
            tmp.append(df[df['id'] == group_num]['abstract'].tolist())
        solu.append(tmp)
    return solu

if __name__=="__main__":
    path = 'D:\\毕设数据\\数据\\event_event.xls'
    group_label_dict = get_grouplabel_bert.get_group_keyword(path)
    path='D:\毕设数据\数据\新监控事件.xlsx'
    feature=new_event_getbert(path)
    tmp=np.loadtxt('text_vectors_new.txt')
    feature_all=np.insert(tmp,0,feature,axis=0)
    np.savetxt('text_vectors_new.txt')
    all_num+=len(feature)
    event_group_num=event_classify(feature,group_label_dict,noise_num)
    solutions=event_solution(event_group_num)
    df = pd.read_excel(path, sheet_name="Sheet1")
    df['solutions']=solutions
    df.to_excel(path,sheet_name="Sheet1")
    noise_per_threshold=0.1
    noise_per=noise_num / all_num
    with open("noise_num.txt", "w", encoding='utf-8') as f:
        f.write(str(all_num)+' '+str(noise_num)+' '+str(noise_per))
    if noise_per>noise_per_threshold:
        print('噪点率过高，需重新聚类')