from sklearn.cluster import KMeans
import numpy as np
from sklearn.cluster import DBSCAN
from sklearn import metrics
from sklearn.manifold import TSNE
import time
import matplotlib.pyplot as plt
import similarity
import get_grouplabel_bert
import clusters_classify



def cluster_center(X):
    kmeans = KMeans(n_clusters=1)
    kmeans.fit(X)
    # 获取质心
    return(kmeans.cluster_centers_)

#计算相似度阈值
def cul_clusters_threshold(center_pos,points):
    min=1
    for i in points:
        s = similarity.cosSim(center_pos[0], i)
        if s<min:
            min=s
    return min

def update_dbscan(min_eps,max_eps,eps_step,min_min_samples,max_min_samples,min_samples_step):
    eps = np.arange(min_eps, max_eps, eps_step)  # eps参数从min_eps开始到max_eps，每隔eps_step进行一次
    min_samples = np.arange(min_min_samples, max_min_samples, min_samples_step)  # min_samples参数从min_min_samples开始到max_min_samples,每隔min_samples_step进行一次
    best_score = 0
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
    return best_score_eps,best_score_min_samples

def cul_culsters_center(point_feature,point_labels):
    # 计算聚类中心
    pos = 0
    label_classify = {}#各聚类中各点向量坐标字典
    for data in point_feature:
        a=point_labels[pos]
        if a not in label_classify:
            label_classify[a] = []
        label_classify[a].append(data)
        pos += 1

    clusters_center = {}#聚类中心向量字典
    clusters_threshold={}#聚类相似度阈值字典
    for label, data in label_classify.items():
        if label != -1:
            center_pos = cluster_center(data)
            threshold=cul_clusters_threshold(center_pos, data)#计算聚类相似度阈值
            clusters_threshold[label]=threshold
            clusters_center[label] = center_pos
    #将所有聚类的相似度阈值写入文件
    with open("clusters_threshold.txt", "w", encoding='utf-8') as f:
        for cluster, threshold in clusters_threshold.items():
            f.writelines(str(cluster) + ':' + str(threshold))
            f.write('\n')

    #将所有聚类的聚类中心向量写入文件
    with open("clusters_center.txt", "w", encoding='utf-8') as f:
        for label, center_pos in clusters_center.items():
            f.writelines(str(label) + ':' + str(center_pos[0].tolist()))
            f.write('\n')
    return clusters_center,label_classify[-1]
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
    #本程序用于对现有数据进行聚类及分类

    feature = np.loadtxt("text_vectors_new.txt")
    #print(feature.shape)

    #eps,min_samples=update_dbscan(0.2,2,0.1,2,10,1)

    #DBSCAN
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
    #记录噪点数
    with open("noise_num.txt", "w", encoding='utf-8') as f:
        f.write(str(len(labels))+' '+str(raito_num)+' '+str(raito))
    #可视化
    '''tsne = TSNE(n_components=2, init='pca', random_state=0)
    #t0 = time()
    data = tsne.fit_transform(feature)
    plot_embedding_2d(data, labels, float(n_clusters_), title=None)'''





    #path = 'D:\\毕设数据\\数据\\event_event.xls'

    #获取所有类别的标签label的分词结果的字典
    #group_label_dict = get_grouplabel_bert.get_group_keyword(path)
    #group_label_bert,maxlen = get_grouplabel_bert.get_bert(group_label_dict)

    #获取所有类别的标签label的分词结果的词向量
    with open('group_label_bert_size.txt', 'r') as read:
        for size in read.readlines():
            size = size.strip('\n')
            size = size.lstrip('(')
            size = size.rstrip(')')
            dim = size.split(', ')
            dim1 = int(dim[0])
            dim2 = int(dim[1])
            dim3 = int(dim[2])
    group_label_bert = np.loadtxt('group_label_bert.txt', delimiter=',').reshape(dim1, dim2, dim3)

    #计算聚类中心或噪点数据向量
    clusters_center,noise_points=cul_culsters_center(feature,labels)
    # 聚类成功相似度衡量
    cluster_group_result,cluster_simi=clusters_classify.attention_get_bert(clusters_center,group_label_bert,True,dim2)

    #聚类不成功相似度衡量
    noise_group_result,noise_simi=clusters_classify.attention_get_bert(noise_points,group_label_bert,False,dim2)

    #收集聚类与噪点的分类结果
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

    #将所有聚类的分类结果写入文件
    with open("clusters_group.txt", "w", encoding='utf-8') as f:
        for cluster, group in cluster_group_result.items():
            f.writelines(str(cluster) + ':' + str(group))
            f.write('\n')

    '''
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
        for j in i:
            L=df1[df1['id']==j]['description'].tolist()
        description.append(L)
    df['description']=description
    #聚类列
    df['cluster_num']=labels
    label = []
    for i in group:
        for j in i:
            L=df2[df2['group'] == j]['label'].tolist()
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

