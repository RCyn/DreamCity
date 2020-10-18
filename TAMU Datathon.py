#!/usr/bin/env python
# coding: utf-8

# In[281]:


###############################################     Libraries      ##############################


# In[6]:


import math
import matplotlib as pp
import numpy as np
import seaborn
from matplotlib import pyplot as plt
import plotly as py
import plotly.graph_objects as go
import plotly.express as px
import datetime as dt
from plotly.subplots import make_subplots
import os
import pandas as pd

import sklearn.datasets as datasets
from sklearn.datasets import make_classification
from sklearn import metrics
from sklearn import cluster, datasets, mixture
from sklearn.neighbors import kneighbors_graph
from sklearn.preprocessing import StandardScaler
from itertools import cycle, islice
from mpl_toolkits.mplot3d import Axes3D


from numpy import where
from numpy import unique

from sklearn.decomposition import PCA
from sklearn.decomposition import KernelPCA


# In[7]:


######################################            Importing and Cleaning              ############################# 


# In[94]:


user_input = [[75,4,5,8,77,88,99,-9999,-9999]]


# In[95]:


df = pd.read_csv(r'C:\Users\wjhwj\Desktop\DreamCity\FinalData.csv')


# In[96]:


df


# In[97]:


col_1='Movehub Rating'
col_2='Purchase Power'
col_3='Health Care'
col_4='Pollution'
col_5='Quality of Life'
col_6='Crime Rating'
col_7='Congestion Level'
col_8='Education'
col_9='Annual Temperature'


# In[98]:


val_df=df.loc[:, col_1:col_9]
val_df


# In[99]:


######################################             End of Importing and Cleaning            ############################# 


# In[100]:


#############################                         Clustering       ###################################################


# In[101]:


X_reduced = PCA(n_components=9).fit_transform(val_df)


# In[102]:


from sklearn.cluster import KMeans
startup = np.array([[60,30,50,30,50,33,22,40,47],[60,60,55,30,80,55,13,35,47],[90,60,55,60,80,33,44,60,70],
           [79,46,66,44,60,41,27,47,29],[90,56,70,56,34,55,33,34,50],[79,56,77,33,55,50,33,55,70],
          [60,46,66,33,52,30,44,55,59],[90,50,45,33,60,41,15,47,70]])
model = KMeans(n_clusters=8, n_init=100, init=startup)
model.fit(X_reduced)
clusters = unique(model.labels_)
df["Cluster"] = model.labels_
print(df.mean(axis = 0))
for i in range(len(user_input[0])):
    if user_input[0][i] == -9999:
        user_input[0][i] = df.mean(axis = 0)[i]
prediction = model.predict(user_input)[0]


# In[103]:


for cluster in clusters:
    row_ix = where(model.labels_ == cluster)
    plt.scatter(X_reduced[row_ix, 0], X_reduced[row_ix, 1])
    if cluster == prediction:
        targetCluster = row_ix
plt.show()


# In[104]:


print(targetCluster)


# In[27]:


######################################             End of Clustering            ############################# 


# In[ ]:





# In[28]:


######################################             Plot Making            ############################# 


# In[29]:


def generate_score(val_df,importance,ideal_temp):
    score=[]
    for i in range(val_df.shape[0]):
        score_num=0
        score_num=score_num+sum([x*y for x,y in zip(val_df.loc[:,col_1:col_8].values.tolist()[i],importance)])
        score_num=score_num+math.fabs(val_df.loc[:,col_9].values.tolist()[i]-ideal_temp)
        score.append(score_num)
    return score


# In[30]:


importance=user_input[0]
for i in range(len(importance)):
    if importance[i]==-9999:
        importance[i]=0
ideal_temp=68
generate_score(val_df,importance,ideal_temp)
df['Score']=generate_score(val_df,importance,ideal_temp)


# In[31]:


#Parse df for ploting
df_to_plot=df.iloc[[targetCluster[0][0]]]

for i in range(1,len(targetCluster[0])):
    df_to_plot=df_to_plot.append(df.iloc[[targetCluster[0][i]]])


# In[32]:


df_to_plot['Score']=generate_score(df_to_plot,importance,ideal_temp)


# In[34]:


df_to_plot


# In[35]:


features = ['Movehub Rating', 'Purchase Power', 'Health Care', 'Quality of Life', 'Pollution', 'Crime Rating','Congestion Level','Education','Annual Temperature']


# In[36]:


def create_graph(df):
    fig = px.scatter_mapbox(df.sort_values('Score', ascending=False).round(),
                        lat="lat", lon="lng", color="Score", hover_name="City",
                        hover_data=features,
                        color_continuous_scale=px.colors.cyclical.IceFire, size_max=15, zoom=1,
                        mapbox_style="carto-positron")
    fig.show()


# In[37]:


create_graph(df_to_plot)


# In[38]:


df.sort_values('Score', ascending=False)[['City', 'Score'] + features].round()


# In[333]:


######################################             End of Plot Making            ############################# 


# In[ ]:





# In[ ]:




