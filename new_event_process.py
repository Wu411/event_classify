from get_keyword_new import load_data,getkeyword
from get_bert import keywords_weight,keywords_vector,self_dict
from get_bert import getbert
import pandas as pd
import similarity
import numpy as np
import tensorflow as tf
from clusters_classify import cul_simlarity


#打开聚类阈值表
with open("clusters_threshold.txt", "r", encoding='utf-8') as f1:
    clusters_threshold = []
    for line in f1.readlines():
        word = line.strip('\n')
        clusters_threshold.append(float(word))

#打开聚类类别对照表
with open("clusters_group.txt", "r", encoding='utf-8') as f2:
    clusters_group = []
    for line in f2.readlines():
        word = line.strip('\n')
        clusters_group.append(int(word))
#打开类别相似度阈值表
with open('group_threshold.txt','r') as f:
    group_threshold=[]
    for i in f.readlines():
        i=i.strip('\n')
        group_threshold.append(float(i))
#打开聚类中心向量表
clusters_center=np.loadtxt("clusters_center.txt")
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
#打开数据总数、噪点数、噪点率表
with open("noise_num.txt", "r", encoding='utf-8') as f4:
    tmp = f4.read()
    tmp = tmp.split(' ')
    all_num = int(tmp[0])
    noise_num = int(tmp[1])
    noise_per = float(tmp[2])

#打开类别标签向量表
groups_label_bert = np.loadtxt('group_label_bert.txt', delimiter = ',').reshape((dim1, dim2, dim3))

#获取新数据词向量
def new_event_getbert(path):
    summary, data = load_data(path,self_dict)  # 读取并处理数据summary
    # 获取每条数据关键词
    res_words = []
    res_weights = []
    res_vector=[]
    for j in summary:
        words, weights,vectors = getkeyword(j, keywords_weight,keywords_vector)
        res_words.append(words)
        res_weights.append(weights)
        res_vector.append(vectors)
    output = getbert(res_words,res_weights,res_vector)
    return output,res_words,data

#噪点数据相似度衡量及分类
def noise_process(noise_point,key_list):
    query = tf.convert_to_tensor(np.asarray(noise_point).reshape(1, 1, 768))
    result = tf.keras.layers.Attention()([query, key_list])
    with tf.Session() as sess:
        result = result.eval()
    groups_result = cul_simlarity(noise_point, result,group_threshold)
    return groups_result

#新数据分类
def event_classify(event_bert,noise_num,res,cluster_id):

    key_list = tf.convert_to_tensor(groups_label_bert)
    for index,group in enumerate(res):
        if group:
            continue
        new=event_bert[index]
        flag=False
        tmp=[]
        tmp1=[]
        label=0
        for center in clusters_center:
            s = similarity.cosSim(np.array(new), np.array(center))
            if s < clusters_threshold[label]:#与聚类中心相似度不符合阈值
                label += 1
                continue
            else:#与聚类中心相似度符合阈值
                flag=True
                tmp.append(tuple(clusters_group[label],s))
                tmp1.append(tuple(label, s))
                label += 1
        if flag==True:#能找到所属聚类
            tmp = tmp.sort(key=lambda x: x[1])
            tmp1 = tmp1.sort(key=lambda x: x[1])
            num=0
            for m,n in zip(tmp,tmp1):
                if num==5:
                    break
                if m not in res[index]:
                    res[index].append(m)
                    cluster_id[index].append(n)
                    num+=1
        else:#未找到所属聚类，按噪点数据分类处理
            noise_num+=1
            #noise_point.append(new)
            #noise_keyword.append(events_keywords[index])
            #noise_summary.append(events_summary[index])
            group_num = noise_process(np.array(new),key_list)
            res[index]=group_num
            cluster_id[index].append(-1)

    return res,cluster_id

def first_classify(event_keywords):
    path='D:\\毕设数据\\数据\\副本train3_增加groupname.xlsx'
    df = pd.read_excel(path, sheet_name="工作表 1 - train")
    data=df[['keyword_new','cluster','group_num']].drop_duplicates()
    keyword_list = data['keyword_new'].values.tolist()
    group_num = data['group_num'].values.tolist()
    cluster = data['cluster'].values.tolist()
    group_res=[]
    cluster_res=[]
    for event in event_keywords:
        group_tmp=[]
        cluster_tmp=[]
        for i,j,z in zip(keyword_list,group_num,cluster):
            if i == str(event):
                group_tmp.append(j)
                cluster_tmp.append(z)
        if group_tmp and cluster_tmp:
            group_res.append(group_tmp)
            cluster_res.append(cluster_tmp)
        else:
            group_res.append([])
            cluster_res.append([])
    return group_res,cluster_res

#获取新事件处理方案
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
    #本程序用于对新数据进行处理

    #获取所有类别的标签label的分词结果的字典
    #path = 'D:\\毕设数据\\数据\\event_event.xls'
    #group_label_dict = get_group_keyword(path)

    #获取新数据词向量并将其加入到现有数据的词向量表中
    path='D:\\毕设数据\\数据\\副本train3_增加groupname.xlsx'
    feature,events_keywords,events_summary=new_event_getbert(path)
    #feature = np.loadtxt("text_vectors_new1.txt")
    #对新数据进行分类并获取分类结果以及对应的处理方法
    new_noise_num=0
    #event_group_num,noise_keyword,noise_summary,new_noise_num=event_classify(feature,new_noise_num,events_keywords,events_summary)
    event_group_num, cluster = first_classify(events_keywords)
    event_group_num, cluster = event_classify(feature, new_noise_num,event_group_num, cluster)
    print(new_noise_num,noise_num)
    noise_num+=new_noise_num
    #solutions=event_solution(event_group_num)
    #将处理方法写入新事件表中
    df = pd.read_excel(path, sheet_name="工作表 1 - train")
    df['group']=event_group_num
    df['cluster'] = cluster
    df.to_excel(path,sheet_name="工作表 1 - train")

    '''with open('noise_point_keywords.txt','a') as f:
        for point_keywords in noise_keyword:
            f.writelines(str(point_keywords))
            f.write('\n')
    with open('noise_point_summary.txt','a') as f:
        for point_summary in noise_summary:
            f.writelines(point_summary)
            f.write('\n')'''
    #获取新的噪点数据数量及噪点率
    all_num = all_num+len(feature)
    noise_per_threshold=0.1#设定噪点率阈值
    noise_per=noise_num / all_num
    with open("noise_num.txt", "w", encoding='utf-8') as f:
        f.write(str(all_num)+' '+str(noise_num)+' '+str(noise_per))
    if noise_per>noise_per_threshold:
        print('噪点率过高，需对所有噪点重新聚类')