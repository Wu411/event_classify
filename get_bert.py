#! -*- coding: utf-8 -*-
# 测试代码可用性: 提取特征
import time
from bert_serving.client import BertClient
import pandas as pd
import numpy as np
import get_keyword_new

config_path = 'chinese_L-12_H-768_A-12/bert_config.json'
checkpoint_path = 'chinese_L-12_H-768_A-12/bert_model.ckpt'
dict_path = 'chinese_L-12_H-768_A-12/vocab.txt'


def load_data(path):
    # 导入需要读取Excel表格的路径
    df = pd.read_excel(path,sheet_name = "Sheet1")
    data=df['keyword_new'].values.tolist()
    res=[]
    for i in data:
        i = i.lstrip('[')
        i = i.rstrip(']')
        i = i.replace('\'', '')
        res.append(i.split(','))
    return res

def getbert(data,vector_name='mean'):
    num=0
    output=[]
    start = time.clock()
    for seg_list in data:
        num += 1
        bc = BertClient()
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
        print('完成提取：', num)
    end = time.clock()
    print('Running time: %s Seconds' % (end - start))
    return output

if __name__ == "__main__":
        

    # 词向量获取方法 cls,mean,
    #vector_name = 'mean'
    #tokenizer = Tokenizer(dict_path, do_lower_case=True)  # 建立分词器
    #model = build_transformer_model(config_path, checkpoint_path)  # 建立模型，加载权重
    #maxlen = 70

    # layer_name = 'Transformer-9-FeedForward-Norm' #获取层的名称
    # intermediate_layer_model = Model(inputs=model.input, 
    #                              outputs=model.get_layer(layer_name).output)#创建的新模型
    #for layers  in model.layers:
    #    print(layers.name)
    #maxlen = 70

    # 读取处理数据
    path='D:\\毕设数据\\数据\\监控事件_202201.xlsx'
    summary, fixkeyword = get_keyword_new.load_data(path)  # 读取并处理数据summary

    # 获取每条数据关键词
    res = []
    for i, j in zip(fixkeyword, summary):
        res.append(get_keyword_new.getkeyword(i, j, get_keyword_new.keywords_dict))
    df = pd.read_excel(path, sheet_name="Sheet1")
    df['keyword_new'] = res
    df.to_excel(path, sheet_name="Sheet1")

    #提取每条数据关键词词向量
    data=load_data(path)
    print('开始提取')
    # 根据提取特征的方法获得词向量
    output=getbert(data)
    print('保存数据')
    np.savetxt("text_vectors_new.txt",output)
