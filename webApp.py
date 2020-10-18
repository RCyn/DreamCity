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
import streamlit as st

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
from PIL import Image
from urllib.request import urlopen

from sklearn.decomposition import PCA
from sklearn.decomposition import KernelPCA


###############       Streamlit App         ###############

logo = Image.open("./DreamCityLogo.png")
st.beta_set_page_config("Your Dream City", logo)

st.title('Your Dream City')

st.write("Find a city for your dream to stay!")
  
image = st.sidebar.image(logo,"",100)

mainTab = st.sidebar.selectbox(
    "Begin your Dream:",
    ("By Preferences", "Feeling lucky?")
)



# In[7]:


######################################            Importing and Cleaning              ############################# 


# In[94]:


# user_input = [[75,4,5,8,77,88,99,-9999,-9999]]


# In[95]:


df = pd.read_csv("https://raw.githubusercontent.com/RCyn/DreamCity/main/FinalData.csv")


# In[96]:


# df


######################     Streamlit setup       #################

default_input = [[-9999]*9]

user_input = default_input
compute = False

if mainTab == "By Preferences":
#  curPrefs = ['Movehub Rating','Purchase Power','Health Care','Pollution','Quality of Life','Crime Rating','Congestion Level','Education','Annual Temperature']
  
  movehubCheck = st.sidebar.checkbox('Movehub Rating')
  if movehubCheck:
    movehub = st.slider(
      'Movehub Rating',
      0, 100, (80))
    st.write('Movehub Rating:', movehub)
    user_input[0][0] = movehub

  purchaseCheck = st.sidebar.checkbox('Purchase Power')
  if purchaseCheck:
    purchase = st.slider(
      'Purchase Power',
      0, 100, (30))
    st.write('Purchase Power:', purchase)
    user_input[0][1] = purchase
  
  healthCheck = st.sidebar.checkbox('Health Care')
  if healthCheck:
    health = st.slider(
      'Health Care',
      0, 100, (70))
    st.write('Health Care:', health)
    user_input[0][2] = health
  
  pollutionCheck = st.sidebar.checkbox('Pollution')
  if pollutionCheck:
    pollution = st.slider(
      'Pollution',
      0, 100, (20))
    st.write('Polution:', pollution)
    user_input[0][3] = pollution
  
  lifeQualityCheck = st.sidebar.checkbox('Quality of Life')
  if lifeQualityCheck:
    lifeQuality = st.slider(
      'Quality of Life',
      0, 100, (90))
    st.write('Quality of Life:', lifeQuality)
    user_input[0][4] = lifeQuality
  
  crimeRatingCheck = st.sidebar.checkbox('Crime Rating')
  if crimeRatingCheck:
    crimeRating = st.slider(
      'Crime Rating',
      0, 100, (10))
    st.write('Crime Rating:', crimeRating)
    user_input[0][5] = crimeRating
  
  congestionCheck = st.sidebar.checkbox('Congestion Level')
  if congestionCheck:
    congestion = st.slider(
      'Congestion Level',
      0, 100, (40))
    st.write('Congestion Level:', congestion)
    user_input[0][6] = congestion
  
  educationCheck = st.sidebar.checkbox('Education')
  if educationCheck:
    education = st.slider(
      'Education',
      0, 100, (90))
    st.write('Education:', education)
    user_input[0][7] = education
  
  climateCheck = st.sidebar.checkbox('Annual Average Temperature (F)')
  if climateCheck:
    climate = st.slider(
      'Annual Average Temperature (F)',
      0, 100, (60))
    st.write('Annual Average Temperature:', climate)
    user_input[0][8] = climate
    
  if not (movehubCheck or purchaseCheck or healthCheck or pollutionCheck or lifeQualityCheck or crimeRatingCheck or congestionCheck or educationCheck or climateCheck):
    st.write('Please select at least one preference from left.')
    
  else:
    # re-run
    compute = st.button("Find")
  
  
else:
  city = st.text_input("Your Dream City is...", "city")
  # re-run
  if city in df["City"].tolist():
    MLcompute = st.button("Find")
  elif city != "city":
    st.write("City not in database, could you provide a different one?")


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


val_df = df.loc[:, col_1:col_9]
# val_df


# In[99]:


######################################             End of Importing and Cleaning            ############################# 


# In[100]:


#############################                         Clustering       ###################################################


# In[101]:


X_reduced = PCA(n_components=9).fit_transform(val_df)


# In[102]:


from sklearn.cluster import KMeans
#startup = np.array([[60,30,50,30,50,33,22,40,47],[60,60,55,30,80,55,13,35,47],[90,60,55,60,80,33,44,60,70],
#           [79,46,66,44,60,41,27,47,29],[90,56,70,56,34,55,33,34,50],[79,56,77,33,55,50,33,55,70],
#          [60,46,66,33,52,30,44,55,59],[90,50,45,33,60,41,15,47,70]])
model = KMeans(n_clusters=8, n_init=100)
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

# st.show(plt)


# In[104]:


# print(targetCluster)


# In[27]:


######################################             End of Clustering            ############################# 


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


# df_to_plot


# In[35]:


features = ['Movehub Rating', 'Purchase Power', 'Health Care', 'Quality of Life', 'Pollution', 'Crime Rating','Congestion Level','Education','Annual Temperature']


# In[36]:


def create_graph(df):
    fig = px.scatter_mapbox(df.sort_values('Score', ascending=False).round(),
                        lat="lat", lon="lng", color="Score", hover_name="City",
                        hover_data=features, size_max=15, zoom=1,
                        mapbox_style="carto-positron")
    st.plotly_chart(fig)


# In[37]:

if mainTab == "By Preferences" and compute:
  create_graph(df_to_plot)


# In[38]:


df.sort_values('Score', ascending=False)[['City', 'Score'] + features].round()


# In[333]:


######################################             End of Plot Making            ############################# 


# In[ ]:





# In[ ]:




