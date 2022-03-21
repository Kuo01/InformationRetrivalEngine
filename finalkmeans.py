# -*- coding: utf-8 -*-
"""
Created on Mon Dec  6 01:32:23 2021

@author: K
"""

import pandas as pd   

import matplotlib.pyplot as plt

from sklearn.feature_extraction.text import TfidfTransformer  
from sklearn.feature_extraction.text import CountVectorizer  


from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer


filePath = 'E:/Master/Fall/InformationRetrievalSystems/Project/testdata/'
saveFilePath = 'E:/Master/Fall/InformationRetrievalSystems/project/allDoc.txt'


kind=[] #存储有多少类文件
labels=[] #用以存储文件名称
corpus=[] #空语料库

summary=[] #保存不同label中每一类文件的个数
numerators=[] #保存每个label中最大的数据值
numerator=0 #最终的purity公式的分子




'''停用词的过滤'''

'''停用词库的建立'''


stop_words = set(stopwords.words('english')) # get stop words
# add single characters into stop words
alph = set([chr(i) for i in range(97, 123)] + [chr(i) for i in range(65, 91)]) 
stop_words = stop_words.union(alph)
print(stop_words)

'''语料库的建立'''

from os import listdir
all_file=listdir(filePath) #Gets all file names in the folder
for i in range(0,len(all_file)):
    #print(i)
    filename=all_file[i]
    filelabel=filename.split('.')[0]
    kind.append(filelabel)
    
    file_add =  filePath + filename
    f = open(file_add,'r',encoding='utf-8')
    sentences=f.readlines()
    f.close()  
    #print(sentences)
    
    for line in sentences:
        #line = sentences[j]
        
        labels.append(filelabel) #文件名称列表
        line = line.split()
        line = [x.lower() for x in line] #单词转化为小写
        #print(line)

        data_adj=''
        for item in line:
            data_adj+=item+' '
        tk = RegexpTokenizer(r"[a-zA-Z]+") # Remove numbers and symbols
        onefile = tk.tokenize(data_adj) #splitting
        data_adj=''
        stemmer = PorterStemmer()
        for item in onefile:
            if item not in stop_words: #Stop word filtering
                data_adj+=stemmer.stem(item)+' ' #Stemming and adding to the string
                
                
        #print(data_adj)
        corpus.append(data_adj) #语料库建立完成
        
        with open(saveFilePath ,"a",encoding='utf-8') as file: 
            file.write(data_adj + "\n")
        
#print(labels)
#print(kind)

#将文本中的词语转换为词频矩阵 矩阵元素a[i][j] 表示j词在i类文本下的词频
vectorizer = CountVectorizer()
 
#该类会统计每个词语的tf-idf权值
transformer = TfidfTransformer()
 
#第一个fit_transform是计算tf-idf 第二个fit_transform是将文本转为词频矩阵
tfidf = transformer.fit_transform(vectorizer.fit_transform(corpus))

#获取词袋模型中的所有词语  
word = vectorizer.get_feature_names()

#将tf-idf矩阵抽取出来，元素w[i][j]表示j词在i类文本中的tf-idf权重
weight = tfidf.toarray()

classNumber=len(kind) #How many types of data have been entered
numerators=[0]*classNumber

print( 'Start Kmeans:')
from sklearn.cluster import KMeans
clf=KMeans(n_clusters=classNumber)
y=clf.fit_predict(weight)
for i in range(0,classNumber):
    summary=[0]*classNumber
    label_i=[]
    for j in range(0,len(y)):
        if y[j]==i:
            label_i.append(labels[j])
    with open('result.txt' ,"a",encoding='utf-8') as file: 
        file.write(str(label_i) + "\n")
    #print('label_'+str(i)+':'+str(label_i))
    
    #Count the number of each type of data in each label
    for k in range(0,len(label_i)):       
        for h in range(0,classNumber):
            if label_i[k]==kind[h]:
                summary[h] = summary[h]+1
    print("label"+ str(i) + ":" )   
    print(summary)
    
    
    for h in range(0,classNumber):
        if numerators[i]<summary[h]:
            numerators[i]=summary[h]
    
    
    
#print(numerators)

for i in range(0,classNumber):
    numerator +=numerators[i]
    
purity = numerator/len(y)

#print(purity)

#每个样本所属的簇
label = []               
i = 1
while i <= len(clf.labels_):
    label.append(clf.labels_[i-1])
    i = i + 1
    
y_pred = clf.labels_

from sklearn.decomposition import PCA
pca = PCA(n_components=2)             #output two dimensional
newData = pca.fit_transform(weight)   #input N dimensional

xs, ys = newData[:, 0], newData[:, 1]
#设置颜色
cluster_colors = {0: 'r', 1: 'purple', 2: 'b', 3: 'chartreuse', 4: 'yellow', 5: '#FFC0CB', 6: '#6A5ACD', 7: '#98FB98'}
              
#设置类名
cluster_names = {0: u'class0', 1: u'class1',2: u'class2',3: u'class3',4: u'class4',5: u'class5',6: u'class6',7: u'class7'} 

df = pd.DataFrame(dict(x=xs, y=ys, label=y_pred, title=corpus)) 
groups = df.groupby('label')

fig, ax = plt.subplots(figsize=(8, 5)) # set size
ax.margins(0.02)
for name, group in groups:
    ax.plot(group.x, group.y, marker='o', linestyle='', ms=5, label=cluster_names[name], color=cluster_colors[name], mec='none')
plt.title(r'$K-Means Clustering$') 

#plt.legend()
plt.text(0.0,0.5,str(kind[0])+"+"+str(kind[1])+"+"+str(kind[2]),fontdict={'size':'16','color':'b'})

plt.show()












