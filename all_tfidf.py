import re
import pandas as pd
from jieba import analyse
import jieba
from urllib import parse
#打开固定关键词文件
with open("fixed_keywords.txt", "r", encoding='utf-8') as f1:
    fixed_keywords = []
    for line in f1.readlines():
        word = line.strip('\n')
        fixed_keywords.append(word)
#打开自定义词典
with open("self_dict.txt", "r", encoding='utf-8') as f2:
    self_dict = []
    for line in f2.readlines():
        word = line.strip('\n')
        self_dict.append(word)
#打开主机名文件
with open("host_name.txt", "r", encoding='utf-8') as f2:
    host_name = []
    for line in f2.readlines():
        word = line.strip('\n')
        host_name.append(word)

def load_data(path):
    # 导入需要读取Excel表格的路径
    df = pd.read_excel(path,sheet_name = "工作表 1 - train")
    #clusters=df['cluster_num'].values.tolist()
    data=df['Summary'].values.tolist()
    summary = []
    for v in data:
        v=parse.unquote(v)#解码
        for name in host_name:
            v=v.replace(name,'hostname')#删除主机名
        for word in fixed_keywords:
            v=v.casefold()
            word=word.casefold()
            if v.find(word)!=-1:
                beg=v.find(word)
                end=beg+len(word)-1
                if beg ==0 and end==len(v)-1:
                    v = v.replace(word, 'fixedkeywords')
                elif beg ==0 and not v[end+1].isalnum() and v[end+1] not in ['_','-']:
                    v=v.replace(word,'fixedkeywords')
                elif end==len(v)-1 and not v[beg-1].isalnum() and v[beg-1] not in ['_','-']:
                    v=v.replace(word,'fixedkeywords')
                elif not v[beg-1].isalnum() and not v[end+1].isalnum() and v[beg-1] not in ['_','-'] and v[end+1] not in ['_','-']:
                    v=v.replace(word,'fixedkeywords')
        v = v.replace("请联系业务岗处理", " ")
        v = v.replace("请联系业务岗确认", " ")
        v = v.replace("需人工介入", " ")
        v = v.replace("联系业务", " ")
        v = v.replace("请联系数据库岗处理", " ")
        v = v.replace("处理", " ")
        v = v.replace("hostname", " ")
        v = v.replace("fixedkeywords", " ")
        k = format_str(v,self_dict)
        k = k.replace("DATE", " ")
        k = k.replace("code", " ")
        k = k.replace("symbol", " ")
        k = k.replace("TIME", " ")
        #k = k.replace("NUMBER", " ")
        k = k.replace("path", " ")
        k = k.replace("url", " ")
        k = k.replace("DOMAIN", " ")
        summary.append(k)
    with open("self_dict.txt", "w", encoding='utf-8') as f:
        for i in self_dict:
            f.writelines(i)
            f.write('\n')
    return summary
def get_word_tfidf(res):
    #jieba.analyse.set_stop_words("stopwords.txt")
    jieba.load_userdict("self_dict.txt")
    tfidf = analyse.extract_tags
    analyse.set_stop_words("stopwords.txt")
    words = tfidf(res,topK=30000,withWeight=True)
    return words
def format_str(txt,self_dict):
    tmp_txt = re.sub(r'\d{4}-\d{1,2}-\d{1,2}', 'DATE', txt)
    tmp_txt = re.sub(r'\d{4}/\d{1,2}/\d{1,2}', 'DATE', tmp_txt)
    tmp_txt = re.sub(r'\d{1,2}/[a-zA-Z]{3}/\d{4}', 'DATE', tmp_txt)
    tmp_txt = re.sub(r'(([0-1]?[0-9])|([2][0-3])):([0-5]?[0-9])(:([0-5]?[0-9]))?', 'TIME', tmp_txt)
    tmp_txt = re.sub(r'((2(5[0-5]|[0-4]\d))|[0-1]?\d{1,2})(\.((2(5[0-5]|[0-4]\d))|[0-1]?\d{1,2})){3}', 'ip', tmp_txt)
    tmp_txt = re.sub(r'/[^.\u4e00-\u9fa5]*\.[^\s\u4e00-\u9fa5]*', 'url', tmp_txt)
    tmp_txt = re.sub(r'\.?(/[^\s\u4e00-\u9fa5]+)+', 'path', tmp_txt)
    tmp_txt = re.sub(r'[a-zA-Z0-9_][-a-zA-Z0-9_]{0,62}(\.[a-zA-Z0-9_][-a-zA-Z0-9_]{0,62})+\.?', 'DOMAIN', tmp_txt)
    tmp_txt = re.sub(r'\w+([-+.]\w+)*@\w+([-.]\w+)*\.\w+([-.]\w+)*', 'Email', tmp_txt)
    update_selfdict(tmp_txt, self_dict)#动态更新自定义词典，加入以下划线和连字符连接的词组
    tmp_txt = re.sub(r'[A-Za-z0-9]{20,}', 'code', tmp_txt)
    #tmp_txt = re.sub(r'(-?\d+)(\.\d+)?', 'NUMBER', tmp_txt)
    tmp_txt = re.sub(r'[^\u4e00-\u9fa5A-Za-z0-9]{2,}', 'symbol', tmp_txt)

    return tmp_txt

def update_selfdict(txt,res):#将下划线和连字符所连固定搭配动态加入自定义词典
    spec_words = re.findall(r'([a-zA-Z][a-zA-Z0-9]+([-_][a-zA-Z0-9]+)+)', txt)
    for i in spec_words:
        if i[0] not in res:
            res.append(i[0])
    return res
path = 'D:\\毕设数据\\数据\\副本train3_增加groupname.xlsx'
#打开固定关键词和自定义固定搭配词典

SUM=load_data(path)#读取并预处理数据summary

#对所有数据的summary进行TF-IDF
string=''
for i in SUM:
    string+=i
    string+='。'
words=get_word_tfidf(string)

result=[]
#去掉分词结果中少于两个字符的英文
for j in words:
    pattern = re.compile(r'((-?\d+)(\.\d+)?)')
    if pattern.match(j[0]):
        continue
    result.append(j)
'''#将固定关键词插入到关键词列表中
for v in fixed_keywords:
    result.insert(0,(v,1))
print(result)'''
#将关键词按权重由大到小写入关键词词典
with open("all_tfidf_dict.txt", "w", encoding='utf-8') as f:
    for i in result:
        f.writelines(str(i))
        f.write('\n')

'''
from collections import defaultdict
import math
import operator
import jieba
from jieba import analyse
import pandas as pd


def loadDataSet():
    with open("stopwords.txt", "r", encoding='utf-8') as f2:
        stopwords = []
        for line in f2.readlines():
            word = line.strip('\n')
            stopwords.append(word)
    df = pd.read_excel('D:\\毕设数据\\数据\\副本train3_增加groupname.xlsx', sheet_name="工作表 1 - train")
    tmp = df['group_name'].dropna().drop_duplicates().values.tolist()
    jieba.load_userdict("self_dict.txt")
    dataset = []
    for w in tmp:
        l=[]
        for j in jieba.lcut(w):
            if j not in stopwords:
                l.append(j)
        dataset.append(l)
    print(dataset)
    return dataset


"""
函数说明：特征选择TF-IDF算法
Parameters:
     list_words:词列表
Returns:
     dict_feature_select:特征选择词字典
"""
#dataset:文件夹，word_list:某一个文件，word某个词

def feature_select(dataset):
    # 总词频统计
    doc_frequency = defaultdict(int) #记录每个词出现的次数，可以把它理解成一个可变长度的list，只要你索引它，它就自动扩列
    for file in dataset:
        for word in file:
            doc_frequency[word] += 1
    # 计算每个词的TF值
    word_tf = {}  # 存储没个词的tf值
    for i in doc_frequency:
        word_tf[i] = doc_frequency[i] / sum(doc_frequency.values()) #sum(doc.frequency.values)

    # 计算每个词的IDF值
    doc_num = len(dataset)
    word_idf = {}  # 存储每个词的idf值
    word_doc = defaultdict(int)  # 存储包含该词的文档数
    for word in doc_frequency:
        for file in dataset:
            if word in file:
                word_doc[word] += 1
    #word_doc和doc_frequency的区别是word_doc存储的是包含这个词的文档数，即如果一个文档里有重复出现一个词则word_doc < doc_frequency
    for word in doc_frequency:
        word_idf[word] = math.log(doc_num / (word_doc[word] + 1))

    # 计算每个词的TF*IDF的值
    word_tf_idf = {}
    for word in doc_frequency:
        word_tf_idf[word] = word_tf[word] * word_idf[word]

    # 对字典按值由大到小排序
    dict_feature_select = sorted(word_tf_idf.items(), key=operator.itemgetter(1), reverse=True)
    return dict_feature_select


if __name__ == '__main__':
    data_list = loadDataSet()  # 加载数据
    features = feature_select(data_list)  # 所有词的TF-IDF值
    with open("new_all_tfidf_dict.txt", "w", encoding='utf-8') as f:
        for i in features:
            f.writelines(i[0]+' '+str(i[1]))
            f.write('\n')'''
