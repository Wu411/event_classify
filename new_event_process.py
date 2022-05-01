import get_keyword_new
import get_bert
import pandas as pd
import similarity
#打开聚类阈值表
with open("clusters_threshold.txt", "r", encoding='utf-8') as f1:
    clusters_threshold = {}
    for line in f1.readlines():
        word = line.strip('\n')
        word=word.split(':')
        key=word[0]
        value=float(word[1])
        clusters_threshold[key]=value
#打开聚类类别对照表
with open("clusters_group.txt", "r", encoding='utf-8') as f2:
    clusters_group = {}
    for line in f2.readlines():
        word = line.strip('\n')
        word=word.split(':')
        key=word[0]
        tmp = word[1].lstrip('[')
        tmp = tmp.rstrip(']')
        value=tmp.split(', ')
        clusters_group[key]=list(map(int,value))
with open("clusters_center.txt", "r", encoding='utf-8') as f3:
    clusters_center = {}
    for line in f3.readlines():
        word = line.strip('\n')
        word=word.split(':')
        key=word[0]
        tmp = word[1].lstrip('[')
        tmp = tmp.rstrip(']')
        value=tmp.split(', ')
        clusters_center[key]=list(map(float,value))
def new_event_getbert(path):
    summary, fixkeyword = get_keyword_new.load_data(path)  # 读取并处理数据summary
    # 获取每条数据关键词
    res = []
    for i, j in zip(fixkeyword, summary):
        res.append(get_keyword_new.getkeyword(i, j, get_keyword_new.keywords_dict))
    output = get_bert.getbert(res)
    return output

def event_classify(event_bert):
    res=[]
    for new in event_bert:
        tmp=[]
        for label,center in clusters_center.items():
            s = similarity.cosSim(new, center)
            if s < clusters_threshold[label]:
                continue
            else:
                for i in clusters_group[label]:
                    if i not in tmp:
                        tmp.append(i)
        res.append(tmp)
    return res

def event_solution(event_group_num):
    path = 'D:\\毕设数据\\数据\\event_solution.xls'
    df = pd.read_excel(path, sheet_name="Sheet1")
    solu=[]
    for event in event_group_num:
        tmp=[]
        for group_num in event:
            tmp.append(df[df['id'] == group_num]['abstract'].tolist()[0])
        solu.append(tmp)
    return solu

if __name__=="__main__":
    path='D:\毕设数据\数据\新监控事件.xlsx'
    df = pd.read_excel(path, sheet_name="Sheet1")
    feature=new_event_getbert(path)
    event_group_num=event_classify(feature)
    solutions=event_solution(event_group_num)
    df['solutions']=solutions
    df.to_excel(path,sheet_name="Sheet1")
