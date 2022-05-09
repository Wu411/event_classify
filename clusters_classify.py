from sklearn.cluster import KMeans
import numpy as np
import similarity
import tensorflow as tf

#注意力机制
def attention(query,key_list):

    result = tf.keras.layers.Attention()([query, key_list])

    return result

def attention_get_bert(clusters_center,key_list,group_threshold,flag):
    #聚类中心与各类别相似度衡量
    if flag==True:
        cluster_group_result = {}#聚类分类结果
        num=0
        for center in clusters_center.values():
            query = tf.convert_to_tensor(np.asarray(center).reshape(1, 1, 768))
            result=attention(query,key_list)#利用注意力机制和中心向量得出的所有的类别代表向量
            with tf.Session() as sess:
                result=result.eval()
            groups_result = cul_simlarity(center[0],result,group_threshold)#相似度衡量
            index = list(clusters_center.keys())[num]
            cluster_group_result[index] = groups_result
            num += 1
        return cluster_group_result
    #噪点数据与各类别相似度衡量
    else:
        noise_group_result = []#噪点分类结果
        noise_simi = []#噪点分类相似度结果
        num=0
        for noise in clusters_center:
            query = tf.convert_to_tensor(np.asarray(noise).reshape(1, 1, 768))
            result = attention(query, key_list)#利用注意力机制和噪点数据向量得出的所有的类别代表向量
            with tf.Session() as sess:
                result = result.eval()
            groups_result = cul_simlarity(noise, result,group_threshold)#相似度衡量
            noise_group_result.append(groups_result)
            #noise_simi.append(simi)
            num+=1
        return noise_group_result

#计算相似度及分类
def cul_simlarity(center,group_bert,group_threshold):
    groups_result={}
    res=[]
    group=0
    for i,j in zip(group_bert,group_threshold):
        s = similarity.cosSim(center, i[0])
        if s >= j:
            groups_result[group]=s
        group += 1
    if groups_result.keys():
        tmp=dict(sorted(groups_result.items(),key=lambda x: x[1]))
        num=0
        for i in tmp.keys():
            num+=1
            if num<=5:
                res.append(i)
            else:
                break
    '''max_score = max(score)
    simi = max_score
    for i, x in enumerate(score):
        if x == max_score:
            group = i
            groups_result.append(group)'''
    return res




