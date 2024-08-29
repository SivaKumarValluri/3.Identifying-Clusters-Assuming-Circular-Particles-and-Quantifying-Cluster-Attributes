"""
Created on Thu May 19 23:53:58 2022

@author: Valluri
"""
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
from sklearn.neighbors import KDTree
import itertools
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN


#Setup_____________________________________________________________________________________________________________________________________________________

os.chdir('C:\\Users\\sivak\\OneDrive - University of Illinois - Urbana\\Desktop\\') #If file not on desktop change file location in data frame excel reader
df = pd.read_excel (r'2wt%HMX.xlsx') #First row is designated as column name by default
particlelocation=np.array(list(zip(df.X, df.Y))) #location in pixels
particleradius=np.array(df["D/pixel"].values/2).reshape(df.shape[0],1)  #Radius of particle in pixels

#Processed image details
h=2748 #Pixels
w=3584 #Pixels

#Current monitor's DPI
my_dpi=96

#Nearest neighbors in a set of radii around the particle___________________________________________________________________________________________________ 

#Dataframe of details
Neighbors_df = pd.DataFrame()
Noofneighbors_df=pd.DataFrame()
radiusinpixel= np.arange(0,50*(df.Pixel[0]/df.micron[0]),(df.Pixel[0]/df.micron[0])) #Radii of influence pursued 
for x in range(0,len(radiusinpixel),1):
    tree = KDTree(particlelocation, leaf_size=2)
    all_nn_indices = tree.query_radius(particlelocation, r=radiusinpixel[x])
    all_nns = [[particlelocation[idx] for idx in nn_indices if idx != i] for i, nn_indices in enumerate(all_nn_indices)]

    listofneighborlocations=[] #List of particle locations seen as neighbors within radius for each particle 
    for nns in all_nns:
        listofneighborlocations.append(nns)
    Neighbors_df[radiusinpixel[x]]=listofneighborlocations
    
    numberofneighbors=[]       #Number of neighbors within defined radius for each particle
    for i in range(0,len(listofneighborlocations),1):
        number=len(listofneighborlocations[i])
        numberofneighbors.append(number) 
    Noofneighbors_df[radiusinpixel[x]]=numberofneighbors
   

from sklearn.neighbors import NearestNeighbors
nearest_neighbors = NearestNeighbors(n_neighbors=11)
neighbors = nearest_neighbors.fit(particlelocation)
distances, indices = neighbors.kneighbors(particlelocation)
distances = np.sort(distances[:,10], axis=0)
distances=distances.reshape(len(particlelocation),1)
numberofparticles=indices[:,0].reshape(len(particlelocation),1)
dydx=np.gradient(distances[:,0],numberofparticles[:,0])
fig, ax1 = plt.subplots()
ax1.set_xlabel('Particles')
ax1.set_ylabel('Distance in Pixel') 
ax1.plot(numberofparticles,distances)
xlim=250#len(particlelocation)
ax1.set_xlim([0, xlim])
ax2 = ax1.twinx()
ax2.plot(numberofparticles,dydx, c='black')
ax2.set_ylim([0, 0.5])
ax2.set_xlim([0, xlim])

#Kneelocator to identify ideal eps (epsiolon) value for DBSCAN clustering__________________________________________________________________________________ 

"""
from kneed import KneeLocator
i = np.arange(len(distances[:,0]))
knee = KneeLocator(i, distances[:,0], S=1, curve='convex', direction='increasing', interp_method='polynomial')
fig = plt.figure(figsize=(5, 5))
knee.plot_knee()
plt.xlabel("No of Particle Centroids")
plt.ylabel("Distance in pixel")
d=distances[knee.knee] #designated as epsilon usually but not in current trial
"""
#Clustering using DBSCAN (better than most clustering algo for such data)_________________________________________________________________________________ 

dbscan_cluster = DBSCAN(eps=40, min_samples=2, metric= 'euclidean') #Parametric space not norrowed yet eps=50 and numberofsamples=8 good starting point
MyDBSCAN=dbscan_cluster.fit(particlelocation)
particle_df = pd.DataFrame(particlelocation)
Particlesinclusters = (particle_df[MyDBSCAN.labels_ != -1]) #-1 labelled points are 'noise' or lone particles
clusterindex=np.array(Particlesinclusters.index)
labels=dbscan_cluster.labels_
noofuniqueelements, noofparticlesincluster = np.unique(labels, return_counts=True)
#Isolating particle radii of particles in clusters
clusterparticlesize=[]
for c in clusterindex:
    clusterparticlesize.append(float(particleradius[c]))

# Number of Clusters
noofuniqueclusters=len(noofuniqueelements)-1 #No of unique elements has clusters as well as isolated particles grouped into 'noise' 
print('Estimated no. of clusters: %d' % noofuniqueclusters)
# Identify Noise
n_noise = list(dbscan_cluster.labels_).count(-1)
print('Estimated no. of noise points: %d' % n_noise)

#Plotting__________________________________________________________________________________________________________________________________________________ 


fig2 = plt.figure(dpi=my_dpi)
plt.figure(figsize=(w/my_dpi, h/my_dpi), dpi=my_dpi)
plt.axis('off')
plt.tick_params(axis='both', left='off', top='off', right='off', bottom='off', labelleft='off', labeltop='off', labelright='off', labelbottom='off')
colors = itertools.cycle(["r", "b", "g"])
plt.scatter(particlelocation[:, 0], particlelocation[:, 1],s= particleradius[:,0]*my_dpi, c='black')
for i in range(0,int(len(Particlesinclusters[0])),1):
    plt.scatter(Particlesinclusters.iloc[i,0], Particlesinclusters.iloc[i,1],s=clusterparticlesize[i]*my_dpi, c='red')
#for i in range(0,int(len(Particlesinclusters[0])),1):
#    plt.scatter(Particlesinclusters.iloc[i,0], Particlesinclusters.iloc[i,1],s=clusterparticlesize[i]*my_dpi, c=next(colors))
plt.xlim([0, w])
plt.ylim([0, h])
plt.show()

#How to choose epsilon and minimum number of samples_______________________________________________________________________________________________________
x=np.arange(1,100,1)
for j in range(10,200,5):
    Effect=[]
    for i in range(1,100,1):
        dbscan_cluster = DBSCAN(eps=j, min_samples=i, metric= 'euclidean')
        MyDBSCAN=dbscan_cluster.fit(particlelocation)
        labels=dbscan_cluster.labels_
        noofuniqueelements, noofparticlesincluster = np.unique(labels, return_counts=True)
        noofuniqueclusters=len(noofuniqueelements)-1 
        Effect.append(noofuniqueclusters)
    y=np.array(Effect, dtype='float')
    plt.loglog(x,y, label=j)
plt.legend(loc="upper right")
plt.xlabel("Min number of particles needed to be considered cluster")
plt.ylabel("No of Clusters")
    