import pandas as pd
import numpy as np
import tensorflow as tf
import similarity
from cluster_texts import cluster_center

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
groups_label_bert = np.loadtxt('group_label_bert.txt', delimiter = ',').reshape((dim1, dim2, dim3))
group_threshold = [1 for i in range(dim1)]
#计算相似度阈值
def cul_clusters_threshold(center_pos,points):
    min=1
    for i in points:
        s = similarity.cosSim(center_pos[0], i)
        if s<min:
            min=s
    return min

#更新新聚类的中心向量及其对应的group
def load_correct_event_classify(path):
    df=pd.read_excel(path, sheet_name='工作表 1 - train')
    cluster_num=df['cluster'].drop_duplicates().values.tolist()#获取更正后的新聚类的分类结果
    # 获取新聚类中心向量
    centers_pos=[]
    clusters_threshold=[]
    cluster_group=[]
    for cluster in cluster_num:
        if cluster==-1:
            continue
        clutser_groups_num=df.loc[df['cluster']==cluster]['group_num'].drop_duplicates().values.tolist()#获取各个新聚类中包含的所有group结果
        for group_num in clutser_groups_num:
            cluster_group.append(int(group_num))
            tmp=df.loc[(df['group_num']==group_num)&(df['cluster']==cluster)]['word_embedding'].values.tolist()#对各个新聚类按照的group重新划分，一个新聚类可能形成多个新聚类

            res=[]
            for i in tmp:
                i = i.lstrip('[')
                i = i.rstrip(']')
                res.append(np.array(list(map(float,i.split(', ')))))
            center_pos = cluster_center(res)#获取重新划分后的各个聚类中心
            centers_pos.append(center_pos[0])
            threshold = cul_clusters_threshold(center_pos, res)  # 计算重新划分后的各个聚类相似度阈值
            clusters_threshold.append(threshold)
    #更新新聚类对应的group
    with open("clusters_group.txt", "a", encoding='utf-8') as f:
        for i in cluster_group:
            f.writelines(str(i))
            f.write('\n')
    #更新聚类中心表
    with open('clusters_center.txt','a') as f:
        np.savetxt(f,centers_pos)
    #更新聚类相似度阈值
    with open('clusters_threshold.txt', 'a') as f:
        for i in clusters_threshold:
            f.write(str(i))
            f.write('\n')
    return centers_pos,cluster_group

def attention(query,key_list):

    result = tf.keras.layers.Attention()([query, key_list])

    return result

def cul_group_threshold(cluster_group,centers,keys_list,group_threshold):
    for i, j in enumerate(centers):
        group = cluster_group[i]
        threshold = group_threshold[group]
        query = tf.convert_to_tensor(np.array(j).reshape(1, 1, 768))
        result = attention(query, tf.convert_to_tensor(
            np.asarray(keys_list[group]).reshape(1, -1, 768)))  # 利用注意力机制和中心向量得出的所有的类别代表向量
        with tf.Session() as sess:
            result = result.eval()
        s = similarity.cosSim(np.array(j), result[0][0])
        if s < threshold:
            group_threshold[group] = s
    return group_threshold

if __name__=="__main__":
    path = 'D:\\毕设数据\\数据\\副本train3_增加groupname.xlsx'
    centers_pos,new_groups_num=load_correct_event_classify(path)
    #key_list=tf.convert_to_tensor(groups_label_bert)
    group_threshold=cul_group_threshold(new_groups_num,centers_pos,groups_label_bert,group_threshold)
    with open('group_threshold.txt','w') as f:
        for i in group_threshold:
            f.writelines(str(i)+'\n')
