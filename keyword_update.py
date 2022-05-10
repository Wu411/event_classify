import pandas as pd
from urllib import parse
import re
import jieba

#打开关键词词典
with open("new_all_tfidf_dict.txt", "r", encoding='utf-8') as f:
    keywords_dict={}
    for line in f.readlines():
        line=line.strip('\n')
        #line = line.lstrip('(')
        #line = line.rstrip(')')
        x=line.split(' ')
        keywords_dict[x[0]]=x[1]



#打开自定义词典
with open("self_dict.txt", "r", encoding='utf-8') as f1:
    self_dict = []
    for line in f1.readlines():
        word = line.strip('\n')
        self_dict.append(word)

def select(data,group_num):
    summary = []
    for v in data:
        v = parse.unquote(v)  # 解码
        v = v.casefold()
        v = v.replace("请联系业务岗处理", " ")
        v = v.replace("请联系业务岗确认", " ")
        v = v.replace("需人工介入", " ")
        v = v.replace("联系业务", " ")
        v = v.replace("请联系数据库岗处理", " ")
        v = v.replace("处理", " ")
        v = v.replace("hostname", " ")
        k = format_str(v)  # 正则表达式处理
        k = k.replace("DATE", " ")
        k = k.replace("code", " ")
        k = k.replace("symbol", " ")
        k = k.replace("TIME", " ")
        k = k.replace("NUMBER", " ")
        k = k.replace("path", " ")
        k = k.replace("url", " ")
        k = k.replace("DOMAIN", " ")
        summary.append(k)
    return summary
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
    tmp_txt = re.sub(r'[A-Za-z0-9]{20,}', 'code', tmp_txt)
    tmp_txt = re.sub(r'(-?\d+)(\.\d+)?', 'NUMBER', tmp_txt)
    tmp_txt = re.sub(r'[^\u4e00-\u9fa5A-Za-z0-9]{2,}', 'symbol', tmp_txt)

    return tmp_txt

def event_keyword(summary_list):
    jieba.load_userdict("self_dict.txt")  # 使用自定义词典
    event=[]
    for string in summary_list:
        seg_list = jieba.lcut(string)
        res = []
        weight = []
        num = 0
        # 选出前十个关键词
        for k, v in keywords_dict.items():
            if num == 10:
                break
            if k.strip('\'') in seg_list and k.strip('\'') not in res:
                weight.append(float(v))
                res.append(k.strip('\''))
                num += 1
        event.append(res)

    return event

def update(path,sheet_name,group_num):
    df = pd.read_excel(path, sheet_name=sheet_name)
    data = df.loc[df['group_num'] == group_num]['Summary'].values.tolist()
    event_summary_list=select(data,group_num)
    event_keywords_list=event_keyword(event_summary_list)
    df1=df.loc[df['group_num'] == group_num]
    df1['keyword_new']=event_keywords_list
    df=df.drop(df[df['group_num'] == group_num].index)
    df = df.append(df1)
    df.to_excel(path,sheet_name=sheet_name)

if __name__=="__main__":
    path = 'D:\\毕设数据\\数据\\副本train3_增加groupname.xlsx'
    sheet_name="工作表 1 - train"
    group_num=3
    update(path,sheet_name,group_num)