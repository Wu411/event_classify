from sklearn.cluster import KMeans
import numpy as np
from sklearn.cluster import DBSCAN
from sklearn import metrics
from sklearn.manifold import TSNE
import time
import matplotlib.pyplot as plt
import similarity

import pandas as pd


def cluster_center(X):
    kmeans = KMeans(n_clusters=1)
    kmeans.fit(np.array(X))
    # 获取质心
    return (kmeans.cluster_centers_)


# 计算相似度阈值
def cul_clusters_threshold(center_pos, points):
    min = 1
    for i in points:
        s = similarity.cosSim(center_pos[0], i)
        if s < min:
            min = s
    return min


def update_dbscan(min_eps, max_eps, eps_step, min_min_samples, max_min_samples, min_samples_step):
    eps = np.arange(min_eps, max_eps, eps_step)  # eps参数从min_eps开始到max_eps，每隔eps_step进行一次
    min_samples = np.arange(min_min_samples, max_min_samples,
                            min_samples_step)  # min_samples参数从min_min_samples开始到max_min_samples,每隔min_samples_step进行一次
    best_score = -2
    best_score_eps = 0
    best_score_min_samples = 0
    for i in eps:
        for j in min_samples:
            try:
                DBS_clf = DBSCAN(eps=i, min_samples=j).fit(feature)
                labels = DBS_clf.labels_  # 和X同一个维度，labels对应索引序号的值 为她所在簇的序号。若簇编号为-1，表示为噪声;
                raito_num = 0
                for v in labels:
                    if v == -1:
                        raito_num += 1
                raito = raito_num / len(labels)
                # labels=-1的个数除以总数，计算噪声点个数占总数的比例
                n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)  # 获取分簇的数目
                k = metrics.silhouette_score(feature, labels)
                score = k - raito
                if score > best_score:
                    best_score = score
                    best_score_eps = i
                    best_score_min_samples = j
            except:
                DBS_clf = ''

    DBS_clf = DBSCAN(eps=best_score_eps, min_samples=best_score_min_samples).fit(feature)
    labels = DBS_clf.labels_  # 和X同一个维度，labels对应索引序号的值 为她所在簇的序号。若簇编号为-1，表示为噪声;
    raito_num = 0
    for v in labels:
        if v == -1:
            raito_num += 1
    raito = raito_num / len(labels)
    # labels=-1的个数除以总数，计算噪声点个数占总数的比例
    print(best_score_eps, best_score_min_samples)
    print('噪声比:', format(raito, '.2%'))
    n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)  # 获取分簇的数目
    print('分簇的数目: %d' % n_clusters_)
    print("轮廓系数: %0.3f" % metrics.silhouette_score(feature, labels))
    return best_score_eps, best_score_min_samples


def cul_culsters_center(point_feature, point_labels):
    # 计算聚类中心
    pos = 0
    label_classify = {}  # 各聚类中各点向量坐标字典
    for data in point_feature:
        a = point_labels[pos]
        if a not in label_classify:
            label_classify[a] = []
        label_classify[a].append(data)
        pos += 1

    clusters_center = {}  # 聚类中心向量字典
    clusters_threshold = {}  # 聚类相似度阈值字典
    for label, data in label_classify.items():
        if label != -1:
            center_pos = cluster_center(data)

            clusters_center[label] = center_pos

    if label_classify[-1]:
        # 将所有聚类的噪点向量写入文件
        np.savetxt("noise_point.txt", label_classify[-1])

    return clusters_center, label_classify[-1]


# 可视化
def plot_embedding_3d(X, target, num, title=None):
    # 坐标缩放到[0,1]区间
    x_min, x_max = np.min(X, axis=0), np.max(X, axis=0)
    X = (X - x_min) / (x_max - x_min)
    # 降维后的坐标为（X[i, 0], X[i, 1],X[i,2]），在该位置画出对应的digits
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1, projection='3d')
    for i in range(X.shape[0]):
        ax.text(X[i, 0], X[i, 1], X[i, 2], str(target[i]),
                color=plt.cm.Set1(target[i] / num),
                fontdict={'weight': 'bold', 'size': 9})
    if title is not None:
        plt.title(title)
    plt.show()


def plot_embedding_2d(data, labels, num, title=None):
    # 坐标缩放到[0,1]区间
    x_min, x_max = np.min(data, 0), np.max(data, 0)
    data = (data - x_min) / (x_max - x_min)

    fig = plt.figure()
    ax = plt.subplot(111)
    for i in range(data.shape[0]):
        plt.text(data[i, 0], data[i, 1], str(labels[i]),
                 color=plt.cm.Set1(labels[i] / num),
                 fontdict={'weight': 'bold', 'size': 5})
    plt.xticks([])
    plt.yticks([])
    plt.show()


if __name__ == "__main__":

    # 本程序用于对现有数据进行聚类及分类

    #feature = np.loadtxt("text_vectors_new1.txt")
    path='D:\\毕设数据\\数据\\副本train3_增加groupname.xlsx'
    df = pd.read_excel(path, sheet_name="工作表 1 - train")
    tmp = df['word_embedding'].values.tolist()  # 对各个新聚类按照的group重新划分，一个新聚类可能形成多个新聚类

    feature = []
    for i in tmp:
        i = i.lstrip('[')
        i = i.rstrip(']')
        feature.append(np.array(list(map(float, i.split(', ')))))
    # eps,min_samples=update_dbscan(0.2,2,0.1,2,10,1)

    # DBSCAN
    # best_score_eps,best_score_min_samples=update_dbscan(0.01,0.2,0.01,2,4,1)
    # with open('dbscan.txt','w') as db:
    #    db.write(str(best_score_eps)+' '+str(best_score_min_samples))
    start = time.clock()
    DBS_clf = DBSCAN(eps=0.01, min_samples=2).fit(feature)
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

    df['cluster'] = labels
    # 记录噪点数
    with open("noise_num.txt", "w", encoding='utf-8') as f:
        f.write(str(len(labels)) + ' ' + str(raito_num) + ' ' + str(raito))
    events_keywords=df.loc[df['cluster']==-1]['keyword_new'].values.tolist()
    events_summary=df.loc[df['cluster']==-1]['Summary'].values.tolist()
    events_feature=df.loc[df['cluster']==-1]['word_embedding'].values.tolist()
    noise_keyword=[]
    noise_summary=[]
    noise_feature=[]
    for i,j,k in zip(events_keywords,events_summary,events_feature):
        i=i.lstrip('[')
        i=i.rstrip(']')
        i=i.split(', ')
        tmp=[]
        for m in i:
            m=m.strip('\'')
            tmp.append(m)
        k=k.lstrip('[')
        k=k.rstrip(']')
        tmp1=k.split(', ')
        noise_keyword.append(tmp)
        noise_summary.append(j)
        noise_feature.append(list(map(float,tmp1)))
    '''with open('noise_point_keywords.txt','w') as f:
        for point_keywords in noise_keyword:
            f.writelines(str(point_keywords))
            f.write('\n')
    with open('noise_point_summary.txt','w') as f:
        for point_summary in noise_summary:
            f.writelines(point_summary)
            f.write('\n')'''
    np.savetxt('noise_point.txt', np.array(noise_feature))

    # 可视化
    '''tsne = TSNE(n_components=2, init='pca', random_state=0)
    #t0 = time()
    data = tsne.fit_transform(feature)
    plot_embedding_2d(data, labels, float(n_clusters_), title=None)'''

    df.to_excel(path, sheet_name="工作表 1 - train")
    #os.system('group_threshold_update.py')
    # 更新新聚类对应的group
    with open("clusters_group.txt", "a", encoding='utf-8') as f:
        f.truncate(0)
    # 更新聚类中心表
    with open('clusters_center.txt', 'a') as f:
        f.truncate(0)
    # 更新聚类相似度阈值
    with open('clusters_threshold.txt', 'a') as f:
        f.truncate(0)


