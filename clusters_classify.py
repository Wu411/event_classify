from sklearn.cluster import KMeans
import numpy as np
import similarity
import tensorflow as tf


def cluster_center(X):
    kmeans = KMeans(n_clusters=1)
    kmeans.fit(X)
    # 获取质心
    return(kmeans.cluster_centers_)

#注意力机制
def attention(query,key_list):

    result = tf.keras.layers.Attention()([query, key_list])

    return result

def attention_get_bert(clusters_center,group_bert,flag,maxlen):
    key_list = tf.convert_to_tensor(np.asarray(group_bert).reshape(-1, maxlen, 768))
    #聚类中心与各类别相似度衡量
    if flag==True:
        cluster_group_result = {}#聚类分类结果
        cluster_simi = {}#聚类分类相似度结果
        num=0
        for center in clusters_center.values():
            query = tf.convert_to_tensor(np.asarray(center).reshape(1, 1, 768))
            result=attention(query,key_list)#利用注意力机制和中心向量得出的所有的类别代表向量
            with tf.Session() as sess:
                result=result.eval()
            groups_result,simi=cul_simlarity(center[0],result)#相似度衡量
            index = list(clusters_center.keys())[num]
            cluster_group_result[index] = groups_result
            cluster_simi[index] = simi
            num += 1
        return cluster_group_result,cluster_simi
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
            groups_result, simi = cul_simlarity(noise, result)#相似度衡量
            noise_group_result.append(groups_result)
            noise_simi.append(simi)
            num+=1
        return noise_group_result,noise_simi

#计算相似度及分类
def cul_simlarity(center,group_bert):
    groups_result=[]
    score=[]
    for i in group_bert:
        s = similarity.cosSim(center, i[0])
        score.append(s)
    max_score = max(score)
    simi = max_score
    for i, x in enumerate(score):
        if x == max_score:
            group = i
            groups_result.append(group)
    return groups_result,simi



