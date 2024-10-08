# PyCirClusterTracker

## Overview ##
This codeset has been used in the published works:
- "Salvati III, L., Valluri, S.K., and Dlott, D.D., From a single grain to microstructure : How explosive grains interact under shock compression. AIP Conf. Proc., 2023. 2844 (1) : 290010."
  
This Python script analyzes the data obtained from segemented images using clustering techniques and generates summary reports in an Excel format. It reads Excel files containing segmented image data (paricle centroid locations and their representative diameter), performs agglomerative clustering assuming circular particles, and outputs detailed results and summaries.

Here is a optical image of particles segmented to identify individual crystals (even SEM micrographs may be used similarly)
  ![image](https://github.com/user-attachments/assets/74c6335d-fe17-4bc7-be4c-243ee5eebd38)

## Module Overview ##

The module includes two functions for analyzing spatial data related to particles and pixel clusters:

1.Agglomerative_Clustering_of_Particles(df, plotting_choice):
   
    - Purpose: Applies DBSCAN clustering to pixel data to identify clusters and visualize results.
    
    - Inputs:
    
        - df: DataFrame with pixel coordinates (X, Y).
        
    - Functionality:
    
        - Analyzes neighbor distances and determines an optimal clustering radius.
        
        - Performs DBSCAN clustering and identifies clusters and noise.
        
        - Provides visualizations of clusters and estimates the number of clusters and noise points.
        
    - Outputs: Clustering results and visualizations.

Here is the image prorgession of initial optical image, the segmented image the data from which (centroid location and respective particle diameter) are passed onto to the code where each particle is assumed to be circular to define clustering.
The colored particles are part of clusters while the black particles are isolated.
![image](https://github.com/user-attachments/assets/a70cde1f-16f1-4a00-b710-5546c03e1965)

The clusters identification and characterstics tracked:
![image](https://github.com/user-attachments/assets/e25af117-a980-4191-95e3-a570b4cff828)

2.DBSCAN_Clustering_of_Emitting_Pixels(df):

    - Purpose: Applies DBSCAN clustering to pixel data to identify clusters and visualize results.
    
    - Inputs:
    
        - df: DataFrame with pixel coordinates (X, Y).
        
    - Functionality:
    
        - Analyzes neighbor distances and determines an optimal clustering radius.
        
        - Performs DBSCAN clustering and identifies clusters and noise.
        
        - Provides visualizations of clusters and estimates the number of clusters and noise points.
        
    - Outputs: Clustering results and visualizations.

In essence, the module helps analyze and visualize clusters in spatial data, whether particles in images or emitting pixels.



## Prerequisites ##

- Python: Ensure Python 3.X is installed on your system.

- Required Libraries: pandas, numpy, glob, os, xlsxwriter (can be installed via pip install pandas numpy xlsxwriter).

- Clustering Script: Ensure that the Clustering_Algorithms module containing Agglomerative_Clustering_of_Particles is available in your working directory.

- Excel workbook containing ImageJ processed csv file that contains centroid locations (X,Y) and representative particle size



## Script Components ##

1.Imports:

    - pandas: Data manipulation and analysis.
    
    - numpy: Numerical operations.
    
    - glob: File path operations.
    
    - os: Operating system interaction.
    
    - xlsxwriter: Excel file writing.

2.User Input:

    - Prompts for the folder address containing the segmented image data.
    
    - Asks if you want to plot fits for each identified cluster.
    
3.Data Processing:

    - Reads Excel files from the specified folder.
    
    - Processes each file using the clustering algorithm.

4.Reporting:

    - Creates an Excel file with summary and detailed cluster data.
    
## Steps to Use the Script ##
    
1.Set Up Your Environment:

    - Ensure all required libraries are installed.
    
    - Place the Clustering_Algorithms module in the working directory.
    
2.Prepare Your Data:

    - Place all Excel files containing segmented image data in a single folder.
    
3.Run the Script:

    - Open a terminal or command prompt (or use Spyder)
    
    -Navigate to the directory where the script is saved.


4.Provide Inputs:

    - Folder Address: Enter the full path to the folder containing the Excel files when prompted.
    
    - Plotting Choice: Respond to whether you want to plot fits for each cluster:
    
    - Type yes or y to plot.
    
    - Type no or n to skip plotting.

5.Script Execution:

    - The script will process each Excel file in the specified folder.
    
    - For each dataset, the script applies agglomerative clustering, calculates metrics, and saves results.

6.Review Results:

    - After processing, an Excel file named folder_name-analysis summary.xlsx will be created.
    
    - The file will contain:
    
        * A sheet named Summary with overall metrics.
        
        * Separate sheets for each dataset with detailed cluster information.

## Excel File Structure ##

1.Summary Sheet:

    - Contains aggregated metrics across all datasets.
    
    - Columns: Total # particles, Smallest size/pixel, Largest size/pixel, Number of Lone particles.

2.Cluster Detail Sheets:

    - Named sequentially (e.g., 1-Clusterdetail, 2-Clusterdetail, etc.).
    
    - Columns: Minor, Major, Aspectratio, No_of_particles_in_cluster.

## Troubleshooting ##

    - Invalid Address: Ensure the folder path is correct and contains Excel files.
    
    - Invalid Input: Ensure the response to the plotting choice is one of the acceptable options (yes, y, no, n).
    
    - Missing Module: Verify that the Clustering_Algorithms module is in the working directory and correctly defined.

## Contact ##

For further assistance, please contact the script author or refer to the documentation of the Clustering_Algorithms module.
