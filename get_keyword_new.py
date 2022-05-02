import re
import pandas as pd
import jieba
from urllib import parse

#打开关键词词典
with open("all_tfidf_dict.txt", "r", encoding='utf-8') as f:
    keywords_dict=[]
    for line in f.readlines():
        line=line.strip('\n')
        x=line.split()
        keywords_dict.append(x[0])

#打开固定关键词文件
with open("fixed_keywords.txt", "r", encoding='utf-8') as f2:
    fixed_words=[]
    for line in f2.readlines():
        word=line.strip('\n')
        fixed_words.append(word)

#打开自定义词典
with open("self_dict.txt", "r", encoding='utf-8') as f1:
    self_dict = []
    for line in f1.readlines():
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
    df = pd.read_excel(path,sheet_name = "Sheet1")
    data=df['SUMMARY'].values.tolist()
    summary = []
    fixkeyword=[]
    for v in data:
        v = parse.unquote(v)#解码
        for name in host_name:
            v = v.replace(name, 'hostname')#替换主机名
        tmp=[]
        for word in fixed_words:
            v=v.casefold()
            word=word.casefold()
            if v.find(word)!=-1:
                beg=v.find(word)
                end=beg+len(word)-1
                if not v[beg-1].isalnum() and not v[end+1].isalnum() and v[beg-1] not in ['_','-'] and v[end+1] not in ['_','-']:
                    v=v.replace(word,' ')
                    tmp.append(word)
        fixkeyword.append(tmp)
        v = v.replace("请联系业务岗处理", " ")
        v = v.replace("请联系业务岗确认", " ")
        v = v.replace("需人工介入", " ")
        v = v.replace("联系业务", " ")
        v = v.replace("请联系数据库岗处理", " ")
        v = v.replace("处理", " ")
        v = v.replace("hostname", " ")
        k = format_str(v)#正则表达式处理
        k = k.replace("DATE", " ")
        k = k.replace("code", " ")
        k = k.replace("symbol", " ")
        k = k.replace("TIME", " ")
        k = k.replace("NUMBER", " ")
        k = k.replace("path", " ")
        k = k.replace("url", " ")
        k = k.replace("DOMAIN", " ")
        summary.append(k)
    #自定义词典动态更新
    with open("self_dict.txt", "w", encoding='utf-8') as f:
        for i in self_dict:
            f.writelines(i)
            f.write('\n')
    return summary,fixkeyword

def format_str(tmp_txt):
    tmp_txt = re.sub(r'\d{4}-\d{1,2}-\d{1,2}', 'DATE', tmp_txt)
    tmp_txt = re.sub(r'\d{4}/\d{1,2}/\d{1,2}', 'DATE', tmp_txt)
    tmp_txt = re.sub(r'\d{1,2}/[a-zA-Z]{3}/\d{4}', 'DATE', tmp_txt)
    tmp_txt = re.sub(r'(([0-1]?[0-9])|([2][0-3])):([0-5]?[0-9])(:([0-5]?[0-9]))?', 'TIME', tmp_txt)
    tmp_txt = re.sub(r'((2(5[0-5]|[0-4]\d))|[0-1]?\d{1,2})(\.((2(5[0-5]|[0-4]\d))|[0-1]?\d{1,2})){3}', 'ip', tmp_txt)
    tmp_txt = re.sub(r'/[^.\u4e00-\u9fa5]*\.[^\s\u4e00-\u9fa5]*', 'url', tmp_txt)
    tmp_txt = re.sub(r'\.?(/[^\s\u4e00-\u9fa5]+)+', 'path', tmp_txt)
    tmp_txt = re.sub(r'[a-zA-Z0-9_][-a-zA-Z0-9_]{0,62}(\.[a-zA-Z0-9_][-a-zA-Z0-9_]{0,62})+\.?', 'DOMAIN', tmp_txt)
    tmp_txt = re.sub(r'\w+([-+.]\w+)*@\w+([-.]\w+)*\.\w+([-.]\w+)*', 'Email', tmp_txt)
    update_selfdict(tmp_txt,self_dict)#动态更新自定义词典，加入以下划线和连字符连接的词组
    tmp_txt = re.sub(r'[A-Za-z0-9]{20,}', 'code', tmp_txt)
    tmp_txt = re.sub(r'(-?\d+)(\.\d+)?', 'NUMBER', tmp_txt)
    tmp_txt = re.sub(r'[^\u4e00-\u9fa5A-Za-z0-9]{2,}', 'symbol', tmp_txt)

    return tmp_txt

def getkeyword(fixkeyword,string,keywords_dict):
    jieba.load_userdict("self_dict.txt")#使用自定义词典
    seg_list=jieba.lcut(string)
    res=[]
    num=0
    #选出前十个关键词
    for i in fixkeyword:
        if i not in res:
            res.append(i)
    for j in keywords_dict:
        if num == 10-len(fixkeyword):
            break
        if j in seg_list and j not in res:
            res.append(j)
            num+=1
    return res

def update_selfdict(txt,res):#将下划线和连字符所连固定搭配动态加入自定义词典
    spec_words = re.findall(r'([a-zA-Z][a-zA-Z0-9]+([-_][a-zA-Z0-9]+)+)', txt)
    for i in spec_words:
        if i[0] not in res:
            res.append(i[0])
    return res

#本程序用于获取关键词