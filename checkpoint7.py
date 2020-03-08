import pandas as pd
from sklearn.cluster import AgglomerativeClustering
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import scipy.cluster.hierarchy as shc

df1=pd.read_csv("CC GENERAL.csv")

df1.drop(["CUST_ID"] ,axis=1,inplace=True)
print(df1.head())
df1.dropna(axis=0,inplace=True,how='any')
model=AgglomerativeClustering(n_clusters=4,affinity='euclidean',linkage="ward")
clust_labels1=model.fit_predict(df1)
algo=pd.DataFrame(clust_labels1)


fig=plt.figure()
ax=fig.add_subplot(111)

scatter=ax.scatter(df1["BALANCE"],df1["PURCHASES"],c=algo[0],s=50)
plt.colorbar(scatter)


plt.figure(figsize=(10,7))




kmeans=KMeans(n_clusters=5,random_state=0)
kmeans.fit(df1)
algo=kmeans.predict(df1)
algo=pd.DataFrame(algo)


kmeans=KMeans(n_clusters=3,random_state=0)
kmeans.fit(df1)
algo=kmeans.predict(df1)
algo=pd.DataFrame(algo)




fig=plt.figure()
ax=fig.add_subplot(111)

scatter=ax.scatter(df1["BALANCE"],df1["PURCHASES"],c=algo[0],s=20)
plt.colorbar(scatter)


plt.figure(figsize=(10,7))

#k=3 is better



