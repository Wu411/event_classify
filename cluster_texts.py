from sklearn.cluster import KMeans
import time
from sklearn.cluster import Birch
from sklearn.externals import joblib
import numpy as np
import pandas as pd
import openpyxl
import random
import json
from sklearn.cluster import DBSCAN
from sklearn.cluster import Birch
from sklearn import metrics
from sklearn.manifold import TSNE
import time
import matplotlib.pyplot as plt
import similarity
import tensorflow as tf
import get_grouplabel_bert
import get_bert
import get_keyword_new
from sklearn.metrics.pairwise import cosine_similarity
import copy


def cluster_center(X):
    kmeans = KMeans(n_clusters=1)
    kmeans.fit(X)
    # 获取质心
    return(kmeans.cluster_centers_)

def attention(query,key_list):

    result = tf.keras.layers.Attention()([query, key_list])

    return result

def attention_get_bert(clusters_center,group_label_dict,group_bert,flag,maxlen):
    key_list = tf.convert_to_tensor(np.asarray(group_bert).reshape(-1, maxlen, 768))
    if flag==True:
        cluster_group_result = {}
        cluster_simi = {}
        num=0
        for center in clusters_center.values():
            query = tf.convert_to_tensor(np.asarray(center).reshape(1, 1, 768))
            result=attention(query,key_list)
            with tf.Session() as sess:
                result=result.eval()
            groups_result,simi=cul_simlarity(center[0],group_label_dict,result)
            index = list(clusters_center.keys())[num]
            cluster_group_result[index] = groups_result
            cluster_simi[index] = simi
            num += 1
        return cluster_group_result,cluster_simi
    else:
        noise_group_result = []
        noise_simi = []
        num=0
        for noise in clusters_center:
            query = tf.convert_to_tensor(np.asarray(noise).reshape(1, 1, 768))
            result = attention(query, key_list)
            with tf.Session() as sess:
                result = result.eval()
            groups_result, simi = cul_simlarity(noise, group_label_dict, result)
            noise_group_result.append(groups_result)
            noise_simi.append(simi)
            num+=1
        return noise_group_result,noise_simi

def cul_simlarity(center,group_label,group_bert):
    '''groups_result={}
    simi={}
    group_num=list(group_label.keys())
    for k,v in center.items():
        score=[]
        groups_result[k]=[]
        for j in group_bert:
            s=similarity.cosSim(v,j)[0]
            score.append(s)
        max_score=max(score)
        simi[k]=max_score
        for i, x in enumerate(score):
            if x == max_score:
                group = group_num[i]
                groups_result[k].append(group)
    return groups_result,simi'''
    groups_result=[]
    group_num = list(group_label.keys())
    score=[]
    for i in group_bert:
        s = similarity.cosSim(center, i[0])
        score.append(s)
    max_score = max(score)
    simi = max_score
    for i, x in enumerate(score):
        if x == max_score:
            group = group_num[i]
            groups_result.append(group)
    return groups_result,simi

def cul_clusters_threshold(center_pos,points):
    min=1
    for i in points:
        s = similarity.cosSim(center_pos[0], i)
        if s<min:
            min=s
    return min

#可视化
def plot_embedding_3d(X, target, num,title=None):
    #坐标缩放到[0,1]区间
    x_min, x_max = np.min(X,axis=0), np.max(X,axis=0)
    X = (X - x_min) / (x_max - x_min)
    #降维后的坐标为（X[i, 0], X[i, 1],X[i,2]），在该位置画出对应的digits
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1, projection='3d')
    for i in range(X.shape[0]):
        ax.text(X[i, 0], X[i, 1], X[i,2],str(target[i]),
                 color=plt.cm.Set1(target[i] / num),
                 fontdict={'weight': 'bold', 'size': 9})
    if title is not None:
        plt.title(title)
    plt.show()

def plot_embedding_2d(data, labels, num,title=None):
    #坐标缩放到[0,1]区间
    x_min, x_max = np.min(data, 0), np.max(data, 0)
    data = (data - x_min) / (x_max - x_min)

    fig = plt.figure()
    ax = plt.subplot(111)
    for i in range(data.shape[0]):
        plt.text(data[i, 0], data[i, 1], str(labels[i]),
                 color=plt.cm.Set1(labels[i]/num),
                 fontdict={'weight': 'bold', 'size': 5})
    plt.xticks([])
    plt.yticks([])
    plt.show()


if __name__ == "__main__":

    path = 'D:\\毕设数据\\数据\\监控事件_202201.xlsx'
    summary, fixkeyword = get_keyword_new.load_data(path)  # 读取并处理数据summary

    # 获取每条数据关键词
    res = []
    for i, j in zip(fixkeyword, summary):
        res.append(get_keyword_new.getkeyword(i, j, get_keyword_new.keywords_dict))
    df = pd.read_excel(path, sheet_name="Sheet1")
    df['keyword_new'] = res
    df.to_excel(path, sheet_name="Sheet1")

    # 根据提取特征的方法获得词向量
    data = get_bert.load_data(path)
    print('开始提取')
    feature = get_bert.getbert(data)
    # 读取提取的特征
    #feature = np.loadtxt("text_vectors_new.txt")
    #print(feature.shape)

    #DBSCAN
    '''eps = np.arange(0.2, 1, 0.1)  # eps参数从0.2开始到4，每隔0.2进行一次
    min_samples = np.arange(2, 4, 1)  # min_samples参数从2开始到20
    best_score = 0
    best_score_eps = 0
    best_score_min_samples = 0
    for i in eps:
        for j in min_samples:
            try:
                DBS_clf = DBSCAN(eps=i, min_samples=j).fit(feature)
                labels = DBS_clf.labels_  # 和X同一个维度，labels对应索引序号的值 为她所在簇的序号。若簇编号为-1，表示为噪声;
                raito_num=0
                for v in labels:
                    if v == -1:
                        raito_num += 1
                raito = raito_num/len(labels)
                # labels=-1的个数除以总数，计算噪声点个数占总数的比例
                n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)  # 获取分簇的数目
                k = metrics.silhouette_score(feature, labels)
                score=k-raito
                if score>best_score:
                    best_score=score
                    best_score_eps=i
                    best_score_min_samples=j
            except:
                DBS_clf=''

    DBS_clf = DBSCAN(eps=best_score_eps, min_samples=best_score_min_samples).fit(feature)
    labels = DBS_clf.labels_  # 和X同一个维度，labels对应索引序号的值 为她所在簇的序号。若簇编号为-1，表示为噪声;
    raito_num = 0
    for v in labels:
        if v == -1:
            raito_num += 1
    raito = raito_num / len(labels)
    # labels=-1的个数除以总数，计算噪声点个数占总数的比例
    print(best_score_eps,best_score_min_samples)
    print('噪声比:', format(raito, '.2%'))
    n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)  # 获取分簇的数目
    print('分簇的数目: %d' % n_clusters_)
    print("轮廓系数: %0.3f" % metrics.silhouette_score(feature, labels))
    '''
    start=time.clock()
    DBS_clf = DBSCAN(eps=0.2, min_samples=2).fit(feature)
    end = time.clock()
    print('Running time: %s Seconds' % (end - start))
    labels = DBS_clf.labels_  # 和X同一个维度，labels对应索引序号的值 为她所在簇的序号。若簇编号为-1，表示为噪声;
    raito_num = 0
    for v in labels:
        if v == -1:
            raito_num += 1
    raito = raito_num / len(labels)
    # labels=-1的个数除以总数，计算噪声点个数占总数的比例
    print('噪声比:', format(raito, '.2%'))
    n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)  # 获取分簇的数目
    print('分簇的数目: %d' % n_clusters_)
    print("轮廓系数: %0.3f" % metrics.silhouette_score(feature, labels))

    #可视化
    '''tsne = TSNE(n_components=2, init='pca', random_state=0)
    #t0 = time()
    data = tsne.fit_transform(feature)
    plot_embedding_2d(data, labels, float(n_clusters_), title=None)'''


    # 计算聚类中心
    pos = 0
    label_classify = {}
    for data in feature:
        a=labels[pos]
        if a not in label_classify:
            label_classify[a] = []
        label_classify[a].append(data)
        pos += 1
    clusters_center = {}
    clusters_threshold={}
    for label, data in label_classify.items():
        if label != -1:
            center_pos = cluster_center(data)
            threshold=cul_clusters_threshold(center_pos, data)
            clusters_threshold[label]=threshold
            clusters_center[label] = center_pos
    with open("clusters_threshold.txt", "w", encoding='utf-8') as f:
        for cluster, threshold in clusters_threshold.items():
            f.writelines(str(cluster) + ':' + str(threshold))
            f.write('\n')
    with open("clusters_center.txt", "w", encoding='utf-8') as f:
        for label, center_pos in clusters_center.items():
            f.writelines(str(label) + ':' + str(center_pos[0].tolist()))
            f.write('\n')
    path = 'D:\\毕设数据\\数据\\event_event.xls'
    group_label_dict = get_grouplabel_bert.get_group_keyword(path)
    group_label_bert,maxlen = get_grouplabel_bert.get_bert(group_label_dict)

    # 聚类成功相似度衡量
    cluster_group_result,cluster_simi=attention_get_bert(clusters_center,group_label_dict,group_label_bert,True,maxlen)

    #聚类不成功相似度衡量
    noise_group_result,noise_simi=attention_get_bert(label_classify[-1],group_label_dict,group_label_bert,False,maxlen)

    event_cluster_result = {}
    similarity_result={}
    pos=0
    for i,j in enumerate(labels):
        if j != -1:
            event_cluster_result[i] = cluster_group_result[j]
            similarity_result[i] = cluster_simi[j]
        else:
            event_cluster_result[i] = noise_group_result[pos]
            similarity_result[i] = noise_simi[pos]
            pos+=1
    with open("clusters_group.txt", "w", encoding='utf-8') as f:
        for cluster, group in cluster_group_result.items():
            f.writelines(str(cluster) + ':' + str(group))
            f.write('\n')

    print('start write')
    path1 = 'D:\\毕设数据\\数据\\监控事件_202201.xlsx'
    path2 = 'D:\\毕设数据\\数据\\event_group.xls'
    path3 = 'D:\\毕设数据\\数据\\event_event.xls'
    df = pd.read_excel(path1, sheet_name="Sheet1")
    df1 = pd.read_excel(path2, sheet_name="event_group")
    df2 = pd.read_excel(path3, sheet_name="Sheet1")
    #类别列
    group=[]
    for k,v in event_cluster_result.items():
        group.append(v)
    df['group']=group
    #相似度列
    score = []
    for k, v in similarity_result.items():
        score.append(v)
    df['cos_similarity']=score
    #类别描述列
    description=[]
    for i in group:
        L=[]
        for j in i:
            L.append(df1[df1['id']==j]['description'].tolist()[0])
        description.append(L)
    df['description']=description
    #聚类列
    df['cluster_num']=labels
    label = []
    for i in group:
        L = []
        for j in i:
            L.append(df2[df2['group'] == j]['label'].tolist())
        label.append(L)
    #类别label列
    df['label']=label
    #类别label列分词结果
    label_cut=[]
    for i in group:
        tmp=[]
        for j in i:
            tmp.append(group_label_dict[j])
        label_cut.append(tmp)
    df['label_cut']=label_cut
    df.to_excel(path1,sheet_name="Sheet1")

