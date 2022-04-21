# ---
短文本聚类、相似度衡量
all_tfidf：用于建立关键词库
get_keywords_new：用于提取未处理事件关键词
get_bert：用于生成关键词词向量
get_grouplabel_bert：用于生成不同group的label词向量
cluster_text：用于对现有监控事件词向量聚类，并计算相似度进行分类
similarity：用于计算相似度
stopwords：停用词表
fixedwords：已确定必须出现的关键词
self_dict：自定义词典
all_tfidf_dict：关键词词典（按权重由大到小排列）
