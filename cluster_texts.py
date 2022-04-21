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
import get_grouplabel_bert
from sklearn.metrics.pairwise import cosine_similarity
import copy


def cluster_center(X):
    kmeans = KMeans(n_clusters=1)
    kmeans.fit(X)
    # 获取质心
    return(kmeans.cluster_centers_)



def cul_simlarity(center,group_label,group_bert):
    groups_result={}
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
    return groups_result,simi

def noise_cul_simlarity(center,group_label,group_bert):
    groups_result={}
    simi={}
    group_num=list(group_label.keys())
    for k,v in enumerate(center):
        score=[]
        groups_result[k] = []
        for j in group_bert:
            s=similarity.cosSim(v,j)
            score.append(s)
        max_score = max(score)
        simi[k] = max_score
        for i, x in enumerate(score):
            if x == max_score:
                group = group_num[i]
                groups_result[k].append(group)
    return groups_result,simi

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
    
    # 读取提取的特征
    feature = np.loadtxt("text_vectors_new.txt")
    #print(feature.shape)

    ''''# k-means 聚类
    model = KMeans(n_clusters = 122, max_iter = 500, random_state = 12)
    kmeans = model.fit(std_data)

    # birth聚类
   
    lists = [10,25]
    best_score = 0
    best_i = -1
    for i in lists:
        print(i)
        y_pred = Birch(branching_factor=i, n_clusters = 9, threshold=0.5,compute_labels=True).fit_predict(feature)
        score = evaluate(y_pred)
        if score > best_score:
            best_score = score
            best_i = i
        print(metrics.calinski_harabaz_score(feature, y_pred)) 
        print(best_score)
        print(best_i)
        '''

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
    labels_dict={}
    for i in labels:
        if i not in labels_dict:
            labels_dict[i]=1
        else:
            labels_dict[i]+=1
    #print(labels_dict)

    #可视化
    tsne = TSNE(n_components=2, init='pca', random_state=0)
    #t0 = time()
    data = tsne.fit_transform(feature)
    plot_embedding_2d(data, labels, float(n_clusters_), title=None)


    '''# 计算聚类中心
    pos = 0
    label_classify = {}
    for data in feature:
        a=labels[pos]
        if a not in label_classify:
            label_classify[a] = []
        label_classify[a].append(data)
        pos += 1
    clusters_center = {}
    for label, data in label_classify.items():
        if label != -1:
            center_pos = cluster_center(data)
            clusters_center[label] = center_pos
    #聚类成功相似度衡量
    path = 'D:\\毕设数据\\数据\\event_event.xls'
    group_label_dict = get_grouplabel_bert.get_group_keyword(path)
    #group_label_bert = get_grouplabel_bert.get_bert(group_label_dict)
    group_label_bert = np.loadtxt("grouplabel_bert.txt")
    group_result,simi=cul_simlarity(clusters_center,group_label_dict,group_label_bert)

    #聚类不成功相似度衡量
    noise_group_result,simi1=noise_cul_simlarity(label_classify[-1],group_label_dict,group_label_bert)
    event_cluster_result = {}
    similarity_result={}
    pos=0
    for i,j in enumerate(labels):
        if j != -1:
            event_cluster_result[i] = group_result[j]
        else:
            event_cluster_result[i] = noise_group_result[pos]
            pos+=1
    pos=0
    for i,j in enumerate(labels):
        if j != -1:
            similarity_result[i] = simi[j]
        else:
            similarity_result[i] = simi1[pos]
            pos+=1

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
    df['group']=group
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
    df.to_excel(path1,sheet_name="Sheet1")'''

