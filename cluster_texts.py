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
import tensorflow as tf
import pandas as pd
import schedule


def cluster_center(X):
    kmeans = KMeans(n_clusters=1)
    kmeans.fit(np.array(X))
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

def update_dbscan(min_eps,max_eps,eps_step,min_min_samples,max_min_samples,min_samples_step,feature):
    eps = np.arange(min_eps, max_eps, eps_step)  # eps参数从min_eps开始到max_eps，每隔eps_step进行一次
    min_samples = np.arange(min_min_samples, max_min_samples, min_samples_step)  # min_samples参数从min_min_samples开始到max_min_samples,每隔min_samples_step进行一次
    best_score = 1
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
                score =  raito #score=k-raito
                if score < best_score: #if score > best_score:
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

            clusters_center[label] = center_pos

    if label_classify[-1]:
        #将所有聚类的噪点向量写入文件
        np.savetxt("noise_point.txt",label_classify[-1])

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

    #feature = np.loadtxt("text_vectors_new1.txt")
    feature = np.loadtxt("noise_point.txt")

    #print(feature.shape)

    #eps,min_samples=update_dbscan(0.2,2,0.1,2,10,1)

    #DBSCAN
    best_score_eps,best_score_min_samples=update_dbscan(0,5,0.1,2,4,1,feature)
    #with open('dbscan.txt','w') as db:
    #    db.write(str(best_score_eps)+' '+str(best_score_min_samples))
    start=time.clock()
    DBS_clf = DBSCAN(eps=best_score_eps, min_samples=best_score_min_samples).fit(feature)
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

    with open('group_threshold.txt','r') as f:
        group_threshold=[]
        for i in f.readlines():
            i=i.strip('\n')
            group_threshold.append(float(i))

    #计算聚类中心或噪点数据向量
    clusters_center,noise_points=cul_culsters_center(feature,labels)
    key_list = tf.convert_to_tensor(np.asarray(group_label_bert).reshape(-1, dim2, 768))
    # 聚类成功相似度衡量
    if clusters_center:
        cluster_group_result=clusters_classify.attention_get_bert(clusters_center,key_list,group_threshold,True)

        noise_keywords=[]
        with open('noise_point_keywords.txt', 'r') as f:
            for line in f.readlines():
                tmp = line.strip('\n')
                tmp = tmp.lstrip('[')
                tmp = tmp.rstrip(']')
                list=tmp.split(', ')
                noise_keywords.append(list)
        noise_summary=[]
        with open('noise_point_summary.txt', 'r') as f:
            for line in f.readlines():
                tmp = line.strip('\n')
                noise_summary.append(tmp)

        keywords=[]
        group=[]
        clusters_id=[]
        word_embedding=[]
        summary=[]
        delete_index=[]
        num=0
        for i,g in cluster_group_result.items():
            num+=1
            for inx,j in enumerate(labels):
                if i==j:
                    delete_index.append(inx)
                    group.append(g)
                    clusters_id.append(num)
                    keywords.append(noise_keywords[inx])
                    word_embedding.append(feature[inx].tolist())
                    summary.append(noise_summary[inx])
        new_noise_keywords=[n for i,n in enumerate(noise_keywords) if i not in delete_index]
        new_noise_summary=[n for i,n in enumerate(noise_summary) if i not in delete_index]
        new_noise_feature=np.delete(feature,delete_index,0)
        df=pd.DataFrame({'cluster':clusters_id,'summary':summary,'keywords':keywords,'group':group,'word_embedding':word_embedding})
        path='D:\\毕设数据\\数据\\new_clusters_group.xlsx'
        df.to_excel(path, sheet_name='Sheet1')
        with open('noise_point_keywords.txt', 'w') as f:
            for point_keywords in new_noise_keywords:
                f.writelines(str(point_keywords))
                f.write('\n')
        with open('noise_point_summary.txt', 'w') as f:
            for point_summary in new_noise_summary:
                f.writelines(point_summary)
                f.write('\n')
        np.savetxt("noise_point.txt",new_noise_feature)
    '''#聚类不成功相似度衡量
    if noise_points:
        noise_group_result=clusters_classify.attention_get_bert(noise_points,key_list,group_threshold,False)
    
    #收集聚类与噪点的分类结果
    event_classify_result = {}
    similarity_result={}
    pos=0
    for i,j in enumerate(labels):
        if j != -1:
            event_classify_result[i] = cluster_group_result[j]
            #similarity_result[i] = cluster_simi[j]
        else:
            event_classify_result[i] = noise_group_result[pos]
            #similarity_result[i] = noise_simi[pos]
            pos+=1
    
    path='D:\\毕设数据\\数据\\副本train3_增加groupname.xlsx'
    df = pd.read_excel(path, sheet_name='工作表 1 - train')
    df['label'] = labels
    df.to_excel(path, sheet_name="工作表 1 - train")


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

