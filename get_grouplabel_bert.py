import pandas as pd
from bert_serving.client import BertClient
import jieba
import numpy as np
import re

#打开自定义词典
with open("stopwords.txt", "r", encoding='utf-8') as f2:
    stopwords = []
    for line in f2.readlines():
        word = line.strip('\n')
        stopwords.append(word)
with open("self_dict.txt", "r", encoding='utf-8') as f2:
    self_dict = []
    for line in f2.readlines():
        word = line.strip('\n')
        self_dict.append(word)
def update_selfdict(path):
    df = pd.read_excel(path, sheet_name="Sheet1")
    data=df['label'].dropna().drop_duplicates().values.tolist()
    for i in data:
        spec_words = re.findall(r'([a-zA-Z][a-zA-Z0-9]+([-_.][a-zA-Z0-9]+)+)', i)
        for i in spec_words:
            if i[0] not in self_dict:
                self_dict.append(i[0])
    with open("self_dict.txt", "w", encoding='utf-8') as f1:
        for i in self_dict:
            f1.writelines(i)
            f1.write('\n')
    return self_dict
def get_group_keyword(path):
    df = pd.read_excel(path, sheet_name="Sheet1")
    data = df.loc[:,['group','label']].dropna()
    group = data['group'].drop_duplicates(keep='last').values.tolist()
    group_keyword={}
    jieba.load_userdict("self_dict.txt")
    for i in group:
        index=int(i)
        group_keyword[index]=[]
        keywords=data.loc[data['group']==i]['label'].values.tolist()
        for j in keywords:
            for k in jieba.lcut(j):
                if k not in stopwords and k not in group_keyword[index]:
                    group_keyword[index].append(k)

    return group_keyword

def get_bert(dict,vector_name='mean'):
    bc = BertClient()
    output=[]
    for seg_list in dict.values():
        if vector_name == 'cls':
            cls_vector = bc.encode(seg_list)[0]
            output.append(cls_vector)
        elif vector_name == 'mean':
            new = []
            vector = bc.encode(seg_list)
            for i in range(768):
                temp = 0
                for j in range(len(vector)):
                    temp += vector[j][i]
                new.append(temp / (len(vector)))
            output.append(new)
    return output

if __name__=='__main__':
    path = 'D:\\毕设数据\\数据\\event_event.xls'
    update_selfdict(path)
    group_label_dict = get_group_keyword(path)
    group_label_bert = get_bert(group_label_dict)
    print('保存数据')
    np.savetxt("grouplabel_bert.txt", group_label_bert)
