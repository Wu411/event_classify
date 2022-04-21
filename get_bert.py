#! -*- coding: utf-8 -*-
# 测试代码可用性: 提取特征
import time
import re
from jieba import analyse
import jieba
#from bert4keras.backend import keras
#from bert4keras.models import build_transformer_model
#from bert4keras.tokenizers import Tokenizer
from bert_serving.client import BertClient
import numpy as np
import pandas as pd
import jieba
import bert
import json
#from keras.models import Model
#from bert4keras.backend import keras, K
from bert4keras.models import build_transformer_model
from bert4keras.tokenizers import Tokenizer
#from bert4keras.optimizers import Adam
#from bert4keras.snippets import sequence_padding, DataGenerator
#from bert4keras.snippets import open
#from bert4keras.layers import ConditionalRandomField
#from keras.layers import Dense
#from keras.models import Model
#from tqdm import tqdm
#from keras.layers import Dropout, Dense

#from keras_bert import extract_embeddings


# os.environ['CUDA_VISIBLE_DEVICES'] = '-1' 

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
    vector_name = 'mean'
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
    data=load_data(path)
    print('开始提取')
    # 根据提取特征的方法获得词向量
    output=getbert(data)
    print('保存数据')
    np.savetxt("text_vectors_new.txt",output)