import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.metrics import confusion_matrix,classification_report

%matplotlib inline

# Get the Data
df = pd.read_csv('College_Data',index_col=0)

# EDA
sns.set_style('whitegrid')
sns.lmplot('Room.Board','Grad.Rate',data=df, hue='Private',palette='coolwarm',size=6,aspect=1,fit_reg=False)
sns.lmplot('Outstate','F.Undergrad',data=df, hue='Private',palette='coolwarm',size=6,aspect=1,fit_reg=False)
sns.set_style('darkgrid')
g = sns.FacetGrid(df,hue="Private",palette='coolwarm',size=6,aspect=2)
g = g.map(plt.hist,'Outstate',bins=20,alpha=0.7)
g = g.map(plt.hist,'Grad.Rate',bins=20,alpha=0.7)
df[df['Grad.Rate'] > 100]

# K-Means-Cluster Creation
kmeans = KMeans(n_clusters=2)
kmeans.fit(df.drop('Private',axis=1))
# K-Means-Cluster Vectors
print(kmeans.cluster_centers_)
# Creating New Column
def converter(cluster):
    if cluster=='Yes':
        return 1
    else:
        return 0
    
print(df['Cluster'] = df['Private'].apply(converter))
print(confusion_matrix(df['Cluster'],kmeans.labels_))
print(classification_report(df['Cluster'],kmeans.labels_))