"""
Created on Sat Jun 18 00:30:31 2022

@author: Siva Kumar Valluri
Set of custom Algorithms to quantify clustering in segmented images
Accept dataframes with centroid coordinates and representative particle diameter
"""
#Accepts dataframe with headings X,Y and D/pixel ((X,Y) is centroid of particle and D is representative particle diameter)
#Requires userinput for decision to plot clusters identified and their fits
def Agglomerative_Clustering_of_Particles(df,plotting_choice):
    import numpy as np
    import matplotlib.pyplot as plt
    from sklearn.neighbors import KDTree
    import matplotlib.pyplot as plt
    import math
    from scipy.spatial import ConvexHull
           
    particlelocation=np.array(list(zip(df.X, df.Y))) #location in pixels
    particleradius=np.array(df["D/pixel"].values/2).reshape(df.shape[0],1)  #Radii of particles in pixels
    Total_number_of_Particles=len(particleradius)
    #Processed image details
    #h=2748 #Pixels
    #w=3584 #Pixels
    
    #Current monitor's DPI
    #my_dpi=96
        
    #Contact Identification--------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    
    def Contactfinder(particlelocation,particleradius):
        #Contact definition: 
        smallest_particle=particleradius.min()
        largest_particle=particleradius.max()
        Threshold_dist=0.1*smallest_particle
        
         
        #Nearest 'k' number of particles distances and their indices
        k=int(math.ceil(2*3.14*particleradius.max()/particleradius.min()))
        tree = KDTree(particlelocation)
        nearest_dist, nearest_indices = tree.query(particlelocation, k=k)
        
        #Converting nearest distances into nearest surface distances to account for particle sizes: Note sufrace distances in micron 
        
        nearest_surfacedist=np.zeros([int(nearest_dist.shape[0]),int(nearest_dist.shape[1])])
        for column in range(0,nearest_dist.shape[1],1):
            for row in range(0,nearest_dist.shape[0],1):    
                nearest_surfacedist[row,column]=nearest_dist[row,column]-particleradius[int(nearest_indices[row,0]),0]-particleradius[int(nearest_indices[row,column]),0]
        
        #nearest_surfacedist=nearest_surfacedist*(int(df.micron[0])/int(df.Pixel[0])) #Converting pixels into micron
        contact_bool=nearest_surfacedist<Threshold_dist
        
        #Identifying number of contact points: Future data point to identify cluster shapes 
        
        No_contactpts=[]
        for row1 in range(0,int(contact_bool.shape[0]),1):
            No_contactpts.append(np.sum(contact_bool[row1,:])-1)            
        No_contactpts=np.array(No_contactpts)
        counter_of_contacts = {}
        for number in No_contactpts:
            if number not in counter_of_contacts:
                counter_of_contacts[number] = 0
            counter_of_contacts[number] += 1            
        
    
        
        #Identifying particles contacting each particle: Prelude to identifying clusters 
        
        List_of_particles_contacting = np.empty(int(nearest_indices.shape[0]), dtype=object)
        for row1 in range(0,int(nearest_indices.shape[0]),1):
            Particle_contacts=[]
            for column1 in range(1,int(nearest_indices.shape[1]),1):
                if contact_bool[row1,column1]==True:
                    Particle_contacts.append(int(nearest_indices[row1,column1]))
            List_of_particles_contacting[row1]=np.array(Particle_contacts, dtype='int')
        List_of_particles_contacting=List_of_particles_contacting.reshape(int(nearest_indices.shape[0]),1)
        
        for element in range(0,int(len(List_of_particles_contacting)),1):
            if len(List_of_particles_contacting[element,0])>0:
                List_of_particles_contacting[element,0]=np.append(List_of_particles_contacting[element,0],element)
        
        return [smallest_particle,largest_particle,counter_of_contacts,List_of_particles_contacting]    
    [smallest_particle,largest_particle,counter_of_contacts,List_of_particles_contacting]=Contactfinder(particlelocation,particleradius)
    print("Number of particles with contact points:")
    print(counter_of_contacts)
   
    #Isolated particles identification: Gives indices and radii of free particles __________________________________________________________________________________________________________________________________________________
    
    def Isolatedparticlefinder(List_of_particles_contacting):
        Isolated_particles_indices=[]
        for element in range(0,int(len(List_of_particles_contacting)),1):
            if len(List_of_particles_contacting[element,0])==0:
                Isolated_particles_indices.append(element)
            
        List_of_particles_contacting=np.delete(List_of_particles_contacting,Isolated_particles_indices)
        List_of_particles_contacting=List_of_particles_contacting.reshape(int(len(List_of_particles_contacting)),1) 
        Isolated_particles_radius=[]
        for i in Isolated_particles_indices:
            Isolated_particles_radius.append(float(particleradius[i]))
        Isolated_particles_radius=np.array(Isolated_particles_radius) #Particle radii are in pixels
        Number_of_Isolated_particles=len(Isolated_particles_indices)
        return [List_of_particles_contacting,Isolated_particles_indices,Isolated_particles_radius,Number_of_Isolated_particles]    
    [List_of_particles_contacting,Isolated_particles_indices,Isolated_particles_radius,Number_of_Isolated_particles]=Isolatedparticlefinder(List_of_particles_contacting)
    print("Number of isolated particles: "+str(len(Isolated_particles_indices))+" | As a fraction of total no. of particles: "+str(len(Isolated_particles_indices)/len(particleradius)))
    
    
    #Cluster Identification: Gives indexes and radii of particles within a cluster aliong with number of particles in cluster_____________________________________________________________________________________________________________________________________________________________
    
    def Clusteridentifier(List_of_particles_contacting):
        
        Listy_particles_contacting=[]#Converting ndarray of numpy objects into list of lists to porcess them as 'sets'
        for i in range(1,len(List_of_particles_contacting),1):
            Listy_particles_contacting.append(list(List_of_particles_contacting[i,0]))
        
        Clusters_indices = []
        while len(Listy_particles_contacting)>0:
            first, *rest = Listy_particles_contacting
            first = set(first)
        
            lf = -1
            while len(first)>lf:
                lf = len(first)
        
                rest2 = []
                for r in rest:
                    if len(first.intersection(set(r)))>0:
                        first |= set(r)
                    else:
                        rest2.append(r)     
                rest = rest2
            Clusters_indices.append(first)
            Listy_particles_contacting = rest
        print("Number of clusters: "+str(len(Clusters_indices)))  
        
        for cluster in range(0,len(Clusters_indices),1): #Converting sets into lists again
            Clusters_indices[cluster]=list(Clusters_indices[cluster])
        
        Clustered_particle_radius=[]
        for cluster in range(0,len(Clusters_indices),1): 
            dummy=list(Clusters_indices[cluster])
            dummy2=[]
            for particleindex in dummy:
                dummyradius=float(particleradius[particleindex])
                dummy2.append(dummyradius)
            Clustered_particle_radius.append(dummy2)
        
        Number_particles_cluster=[]
        for cluster in Clusters_indices:
            number=int(len(cluster))
            Number_particles_cluster.append(number)
        
        Number_particles_in_cluster=np.array(Number_particles_cluster)        
        return [Clusters_indices,Clustered_particle_radius,Number_particles_in_cluster]
    
    [Clusters_indices,Clustered_particle_radius,Number_particles_in_cluster]=Clusteridentifier(List_of_particles_contacting)
    
    #Cluster bounding box shape factors_________________________________________________________________________________________________________________________________________________________________________________________________________
    
    def getcircumferencepoints(radius,center_x,center_y,number_of_points=50):
        radians_between_each_point = 2*np.pi/number_of_points
        list_of_points = []
        for p in range(0, number_of_points):
            list_of_points.append( [radius*np.cos(p*radians_between_each_point)+ center_x,radius*np.sin(p*radians_between_each_point)+ center_y])
        return list_of_points
    
    def Clusterfitfinder(Clusters_indices,plotting_choice):  
        List_of_min_area_rect_fits=[]
        List_of_axis=np.zeros((len(Clusters_indices),2))
        clustercounter=1
        for cluster in Clusters_indices:
            circumference_points = np.zeros([0,2])  
            for particle in range(len(cluster)):
                center_x=df.X.iloc[cluster[particle]]
                center_y=df.Y.iloc[cluster[particle]]
                radius=df.iloc[cluster[particle]][4]/2   
                number_of_points= int(round(3*radius))
                list_of_points=np.array(getcircumferencepoints(radius,center_x,center_y,number_of_points))
                circumference_points = np.concatenate((circumference_points, list_of_points), axis=0) 
            Hull=ConvexHull(circumference_points)
            pi2 = np.pi/2.
            hull_points = circumference_points[Hull.vertices]
            
            # calculate edge angles
            edges = np.zeros((len(hull_points)-1, 2))
            edges = hull_points[1:] - hull_points[:-1]
    
            angles = np.zeros((len(edges)))
            angles = np.arctan2(edges[:, 1], edges[:, 0])
    
            angles = np.abs(np.mod(angles, pi2))
            angles = np.unique(angles)
            
            rotations = np.vstack([np.cos(angles),np.cos(angles-pi2),np.cos(angles+pi2),np.cos(angles)]).T
            rotations = rotations.reshape((-1, 2, 2))
            
            rot_points = np.dot(rotations, hull_points.T)
    
            #find the bounding points
            min_x = np.nanmin(rot_points[:, 0], axis=1)
            max_x = np.nanmax(rot_points[:, 0], axis=1)
            min_y = np.nanmin(rot_points[:, 1], axis=1)
            max_y = np.nanmax(rot_points[:, 1], axis=1)
            
            areas = (max_x - min_x) * (max_y - min_y)
            best_idx = np.argmin(areas)
    
            #return the best box
            x1 = max_x[best_idx]
            x2 = min_x[best_idx]
            y1 = max_y[best_idx]
            y2 = min_y[best_idx]
            r = rotations[best_idx]
            
            #Corners of rectangle fit
            rectangle_corner = np.zeros((4, 2))
            rectangle_corner[0] = np.dot([x1, y2], r)
            rectangle_corner[1] = np.dot([x2, y2], r)
            rectangle_corner[2] = np.dot([x2, y1], r)
            rectangle_corner[3] = np.dot([x1, y1], r)
            
            #Sides of rectangle
            dist1 = np.linalg.norm(rectangle_corner[0] - rectangle_corner[1])
            dist2 = np.linalg.norm(rectangle_corner[1] - rectangle_corner[2])
            dist3 = np.linalg.norm(rectangle_corner[2] - rectangle_corner[3])
            dist4 = np.linalg.norm(rectangle_corner[3] - rectangle_corner[0])
            
            
            List_of_min_area_rect_fits.append(rectangle_corner)
            List_of_axis[clustercounter-1,0]=min(dist1,dist2,dist3,dist4)
            List_of_axis[clustercounter-1,1]=max(dist1,dist2,dist3,dist4)
            
            #Plotting perimeters of representative particles in cluster, convexhull and minimum area bounding rectangle
            #Plots only if user 
            if plotting_choice.lower() in  ["y","yes","yippee ki yay","alright","alrighty"]:
                plt.scatter(circumference_points[:,0], circumference_points[:,1],s=5, c='red', marker='o')
                for simplex in Hull.simplices:
                    plt.plot(circumference_points[simplex,0], circumference_points[simplex,1], 'k-')        
                plt.fill(rectangle_corner[:,0], rectangle_corner[:,1], alpha=0.2)
                plt.axis('equal')
                plt.title('Cluster no.'+str(clustercounter))
                plt.xlabel('x/pixels') 
                plt.ylabel('y/pixels')
                plt.show()
            
            clustercounter=clustercounter+1
        return [List_of_min_area_rect_fits,List_of_axis]
    
    List_of_min_area_rect_fits,List_of_axis=Clusterfitfinder(Clusters_indices,plotting_choice)
    
    return [Total_number_of_Particles,smallest_particle,largest_particle,counter_of_contacts,Number_particles_in_cluster,Number_of_Isolated_particles,List_of_axis]



#Accepts dataframe with headings X,Y  (Emitting pixels coordinates)
def DBSCAN_Clustering_of_Emitting_Pixels(df):
    import numpy as np
    import matplotlib.pyplot as plt
    import pandas as pd
    import os
    from sklearn.neighbors import KDTree
    import matplotlib.pyplot as plt
    from sklearn.cluster import DBSCAN
    
    
    #os.chdir('C:\\Users\\sivak\\Desktop') #If file not on desktop change file location in data frame excel reader
    #df = pd.read_excel (r'10_small.xlsx') #First row is designated as column name by default
    particlelocation=np.array(list(zip(df.X, df.Y))) #location in pixels
    #particleradius=np.array(df["D/pixel"].values/2).reshape(df.shape[0],1)  #Radius of particle in pixels
    
    
    #Nearest neighbors in a set of radii around the particle 
    #Dataframe of details
    Neighbors_df = pd.DataFrame()
    Noofneighbors_df=pd.DataFrame()
    radiusinpixel= np.arange(0,50,1) #Radii of influence pursued 
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
    
    #Kneelocator to identify ideal eps (epsiolon) value for DBSCAN clustering 
    from kneed import KneeLocator
    i = np.arange(len(distances[:,0]))
    knee = KneeLocator(i, distances[:,0], S=1, curve='convex', direction='increasing', interp_method='polynomial')
    fig = plt.figure(figsize=(5, 5))
    knee.plot_knee()
    plt.xlabel("No of Particle Centroids")
    plt.ylabel("Distance in pixel")
    d=distances[knee.knee] #designated as epsilon usually but not in current trial
    
    #Clustering using DBSCAN (bettern than most clustering algo for such data) 
    dbscan_cluster = DBSCAN(eps=45, min_samples=2, metric= 'euclidean') #Parametric space not norrowed yet eps=50 and numberofsamples=8 good starting point
    MyDBSCN=dbscan_cluster.fit(particlelocation)
    particle_df = pd.DataFrame(particlelocation)
    clustersidentified = (particle_df[MyDBSCN.labels_ != -1]) #-1 labelled points are 'noise' or lone particles
    clusterindex=np.array(clustersidentified.index)
    
    #Isolating particle radii of particles in clusters

        
    #Plotting clusters
    #color_list = ['green', 'blue', 'red', 'yellow',  'orange',  'magenta', 'cyan', 'purple']
    fig2 = plt.figure(dpi=144)
    #plt.scatter(particlelocation[:, 0], particlelocation[:, 1],s= particleradius[:,0]*(df.Pixel[0]/df.micron[0]), c='black')
    plt.scatter(np.array(clustersidentified[0]), np.array(clustersidentified[1]),s=1, c='red')
    plt.xlabel("y in pixels")
    plt.ylabel("x in pixels")
    
    # Number of Clusters
    labels=dbscan_cluster.labels_
    N_clus=len(set(labels))-(1 if -1 in labels else 0)
    print('Estimated no. of clusters: %d' % N_clus)
    
    # Identify Noise
    n_noise = list(dbscan_cluster.labels_).count(-1)
    print('Estimated no. of noise points: %d' % n_noise)
    # Calculating v_measure
    #print('v_measure =', v_measure_score(y, labels))
    return [radiusinpixel,Noofneighbors_df] # Currently just radius of influence and number of neighbors within it.
#Add more based on what's needed 

