from get_keyword_new import load_data,getkeyword,keywords_dict
from get_bert import getbert
import pandas as pd
import similarity
import numpy as np
import tensorflow as tf
from get_grouplabel_bert import get_group_keyword
from clusters_classify import cul_simlarity


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
    summary, fixkeyword = load_data(path)  # 读取并处理数据summary
    # 获取每条数据关键词
    res = []
    for i, j in zip(fixkeyword, summary):
        res.append(getkeyword(i, j, keywords_dict))
    output = getbert(res)
    return output

#噪点数据相似度衡量及分类
def noise_process(noise_point):
    key_list = tf.convert_to_tensor(groups_label_bert)
    query = tf.convert_to_tensor(np.asarray(noise_point).reshape(1, 1, 768))
    result = tf.keras.layers.Attention()([query, key_list])
    with tf.Session() as sess:
        result = result.eval()
    groups_result, simi = cul_simlarity(noise_point, result)
    return groups_result,simi

#新数据分类
def event_classify(event_bert,noise_num):
    res=[]
    for new in event_bert:
        flag=False
        tmp=[]
        for label,center in clusters_center.items():
            s = similarity.cosSim(new, center)
            if s < clusters_threshold[label]:#与聚类中心相似度不符合阈值
                continue
            else:#与聚类中心相似度符合阈值
                flag=True
                for i in clusters_group[label]:#获取该聚类对应的类别结果
                    if i not in tmp:
                        tmp.append(i)
        if flag==True:#能找到所属聚类
            res.append(tmp)
        else:#未找到所属聚类，按噪点数据分类处理
            noise_num+=1
            group_num, simi = noise_process(new)
            res.append(group_num)
    return res

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
    path='D:\毕设数据\数据\新监控事件.xlsx'
    feature=new_event_getbert(path)
    tmp=np.loadtxt('text_vectors_new.txt')
    feature_all=np.insert(tmp,0,feature,axis=0)
    np.savetxt('text_vectors_new.txt')

    #对新数据进行分类并获取分类结果以及对应的处理方法
    event_group_num=event_classify(feature,noise_num)
    solutions=event_solution(event_group_num)
    #将处理方法写入新事件表中
    df = pd.read_excel(path, sheet_name="Sheet1")
    df['solutions']=solutions
    df.to_excel(path,sheet_name="Sheet1")

    #获取新的噪点数据数量及噪点率
    all_num = len(feature_all)
    noise_per_threshold=0.1#设定噪点率阈值
    noise_per=noise_num / all_num
    with open("noise_num.txt", "w", encoding='utf-8') as f:
        f.write(str(all_num)+' '+str(noise_num)+' '+str(noise_per))
    if noise_per>noise_per_threshold:
        print('噪点率过高，需重新聚类')