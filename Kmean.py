import numpy as np
import sklearn.datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler,StandardScaler
import pandas as pd
import math,random
from skimage import io
import cv2

class KMeans:
  def __init__(self):
    self.dim=0
    self.cluster_centers=None
    self.cluster_index=None

  def set_initial_random_state(self,data):
    self.cluster_centers=np.array(random.choices(data,k=self.n_clusters))

  def nearest_cluster(self,data,clusters_centers):
    nearest=np.array(np.zeros(data.shape[0],))
    for i in range(0,data.shape[0]):
      error=10e12
      for j in range(0,clusters_centers.shape[0]):
        if error> np.linalg.norm(data[i]-clusters_centers[j]):
          error=np.linalg.norm(data[i]-clusters_centers[j])
          nearest[i]=j
    return nearest

  def mean_error(self,data,clusters_index):
    error=0
    for i in range(0,data.shape[0]):
        error+=np.linalg.norm(data[i]-self.cluster_centers[clusters_index[i]])
    return error/self.n_clusters

  def fit(self,data,clusters,iterate=1,show_log=True):
    self.dim=data.shape[1]
    self.n_clusters=clusters
    self.set_initial_random_state(data)
    last_error=10e12
    for i in range(0,iterate):
      self.cluster_index=self.nearest_cluster(data,self.cluster_centers).astype(int)
      for j in range(0,self.n_clusters):
          self.cluster_centers[j]=data[self.cluster_index==j].mean(axis=0)
      error=self.mean_error(data,self.cluster_index)
      if(error==last_error):
        break
      last_error=error
      if show_log:
        print("Error at "+str(i) + "th iteration : "+str(error))
    return last_error

  def find_optimize_cluster_K(self,data):
    error=[]
    x=range(1,data.shape[0]+1)
    for i in x:
      error.append(self.fit(data,i,show_log=False))
    df=pd.DataFrame(data=error,index=x)
    print("Desired K : "+str((df.shift(1)-df).idxmax()[0]+1))
    df.plot()
    
    
#reduce size of image

image = cv2.imread('image.png').astype(np.int32)
rows = image.shape[0]
cols = image.shape[1]
image = image.reshape(rows*cols, 3)

Clustering=KMeans()
Clustering.fit(image,64,iterate=2)

image=Clustering.cluster_centers[Clustering.cluster_index]
image=image.reshape(rows,cols, 3)

cv2.imwrite("new_image.png", image)