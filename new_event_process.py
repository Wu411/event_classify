from get_keyword_new import load_data,getkeyword
from get_bert import keywords_dict
from get_bert import getbert
import pandas as pd
import similarity
import numpy as np
import tensorflow as tf
from get_grouplabel_bert import get_group_keyword
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
        tmp = word.lstrip('[')
        tmp = tmp.rstrip(']')
        value=tmp.split(', ')
        clusters_group.append(list(map(int,value)))

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
    summary, fixkeyword = load_data(path)  # 读取并处理数据summary
    # 获取每条数据关键词
    res_words = []
    res_weights = []
    for j in summary:
        words, weights = getkeyword(j, keywords_dict)
        res_words.append(words)
        res_weights.append(weights)
    output = getbert(res_words,res_weights)
    return output,res_words

#噪点数据相似度衡量及分类
def noise_process(noise_point,key_list):
    query = tf.convert_to_tensor(np.asarray(noise_point).reshape(1, 1, 768))
    result = tf.keras.layers.Attention()([query, key_list])
    with tf.Session() as sess:
        result = result.eval()
    groups_result = cul_simlarity(noise_point, result)
    return groups_result

#新数据分类
def event_classify(event_bert,noise_num,events_keywords):
    res=[]
    noise_point=[]
    noise_keyword=[]
    key_list = tf.convert_to_tensor(groups_label_bert)
    for index,new in enumerate(event_bert):
        flag=False
        tmp=[]
        label=0
        for center in clusters_center:
            s = similarity.cosSim(new, center)
            if s < clusters_threshold[label]:#与聚类中心相似度不符合阈值
                label += 1
                continue
            else:#与聚类中心相似度符合阈值
                flag=True
                for i in clusters_group[label]:#获取该聚类对应的类别结果
                    if i not in tmp:
                        tmp.append(i)
                label += 1
        if flag==True:#能找到所属聚类
            res.append(tmp)
        else:#未找到所属聚类，按噪点数据分类处理
            noise_num+=1
            noise_point.append(new)
            noise_keyword.append(events_keywords[index])
            group_num = noise_process(new,key_list)
            res.append(group_num)
    if noise_point:
        print('保存噪点数据')
        tmp = np.loadtxt('noise_point.txt')
        feature_all = np.insert(tmp, 0, noise_point, axis=0)
        np.savetxt('noise_point.txt',feature_all)

    return res,noise_keyword

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
    feature,events_keywords=new_event_getbert(path)

    #对新数据进行分类并获取分类结果以及对应的处理方法
    event_group_num,noise_keyword=event_classify(feature,noise_num,events_keywords)
    solutions=event_solution(event_group_num)
    #将处理方法写入新事件表中
    df = pd.read_excel(path, sheet_name="工作表 1 - train")
    df['solutions']=solutions
    df.to_excel(path,sheet_name="工作表 1 - train")

    with open('noise_point_keywords.txt','a') as f:
        for point_keywords in noise_keyword:
            f.writelines(str(point_keywords))
            f.write('\n')
    #获取新的噪点数据数量及噪点率
    all_num = all_num+len(feature)
    noise_per_threshold=0.1#设定噪点率阈值
    noise_per=noise_num / all_num
    with open("noise_num.txt", "w", encoding='utf-8') as f:
        f.write(str(all_num)+' '+str(noise_num)+' '+str(noise_per))
    if noise_per>noise_per_threshold:
        print('噪点率过高，需对所有噪点重新聚类')