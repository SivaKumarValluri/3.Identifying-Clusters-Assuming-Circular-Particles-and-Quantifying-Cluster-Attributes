"""
Created on Fri Jun 17 20:43:53 2022

@author: Siva Kumar Valluri
"""
import pandas as pd
import os
import glob 
import numpy as np

os.chdir('C:\\Users\\sivak\\.spyder-py3') #Directory of scripts
address = input("Enter address of folder with Segmented images (just copy paste address): ")
while True:
    plotting_choice = input("Do you want to plot fit for each cluster identified? Yes/No : ").lower() 
    try:
        if plotting_choice.lower() in  ["y","yes","yippee ki yay","alright","alrighty"]:
            print("Suit yourself! It's slow now thanks to your choice")
        elif plotting_choice.lower() in ["n","no","nope"]:
            print("Good for you")
        else:
            raise Exception("Invalid input! Answer can only be 'Yes' or 'No'")
    except Exception as e:
        print(e)    
    else:
        break

directory_of_details=[]
for excelfile in glob.glob(str(address)+"/*.xlsx"):
    df = pd.read_excel (excelfile)
    directory_of_details.append(df)

from Clustering_Algorithms import Agglomerative_Clustering_of_Particles 
folder_name=address.rpartition('\\')[2]

Excelwriter = pd.ExcelWriter(str(folder_name)+'-analysis summary.xlsx', engine='xlsxwriter')

Dsummary=pd.DataFrame()
counter=1
for dataframe in directory_of_details:
    Total_number_of_Particles,smallest_particle,largest_particle,counter_of_contacts,Number_particles_in_cluster,Number_of_Isolated_particles,List_of_axis=Agglomerative_Clustering_of_Particles(dataframe,plotting_choice)
    Dataset = np.column_stack((Total_number_of_Particles,smallest_particle,largest_particle,Number_of_Isolated_particles))    
    X = pd.DataFrame(Dataset,columns = ['Total # particles','Smallest size/pixel','Largest size/pixel','Number of Lone particles'])
    Dsummary = Dsummary.append(X)
    
    Dcluster=pd.DataFrame()
    Dcluster=Dcluster.assign(Minor=List_of_axis[:,0],Major=List_of_axis[:,1],Aspectratio=List_of_axis[:,1]/List_of_axis[:,0],No_of_particles_in_cluster = Number_particles_in_cluster)
    Dcluster.to_excel(Excelwriter, sheet_name=str(counter)+'-Clusterdetail',index=False) 
        
    counter=counter+1
  
Dsummary.to_excel(Excelwriter, sheet_name='Summary',index=False) 
Excelwriter.save()
Excelwriter.close()