##-------------------------------------------------------------------------------------------------------------
# Student Name : Harshita Patil
# Code for Question 1
# Description: A program, to apply k means clustering in data set provided below:
# Data Set: CC.CSV
#It will also
#  i) Remove all null values by the mean.
# ii) Use of the elbow method to find a good number of clusters with KMeans Algorithm.
# This program also calculates the silhouette score for the above clustering and trying feature scaling
# to see if it will improve the Silhouette score. Applying PCA on the same dataset.
##-------------------------------------------------------------------------------------------------------------
# Imported the necessary libraries and created our environment.
# Importing pandas library as we will be doing dataframe manipulation.
import pandas as pd
# Importing pandas library as we will be doing dataframe manipulation.
import matplotlib.pyplot as plt
# Importing Principal Component Analaysis(PCA) which is a linear dimensionality reduction technique.
from sklearn.decomposition import PCA
# seaborn uses matplotlib to draw its plots.
import seaborn as sns

# Next is importing ‘CC.csv’ file into dataframe.
# We are using pd.read_csv() method of Pandas for that. It creates a Dataframe from a csv file.
dataset = pd.read_csv('CC.csv')
# Printing the number of rows and column in Data set
print('The number of rows and column in Dataset:', dataset.shape,'\n')

# to drop the CUST_ID column
dataset = dataset.drop('CUST_ID', axis=1)


# Check total number of null values
print("Total number of null values:  ", + dataset.isnull().sum().sum(),'\n')

# Replace all the null with mean
dataset.fillna(dataset.mean(), inplace=True)
print("Total number of null after mean replace:  ", + dataset.isnull().sum().sum(),'\n')

from sklearn.cluster import KMeans
# Creating elbow method to get optimal number of Clusters
def elb_met(eldf):
# Calculating wcss for a range of values for k
        wcss = []
# using for loop in range (1,10)
        for i in range(1,10):
            kmeans = KMeans(n_clusters=i, init='k-means++', max_iter=300, random_state=42)
            kmeans.fit(eldf)
            wcss.append(kmeans.inertia_)

# provided range (1,10)
        plt.plot(range(1,10),wcss)
# Set tittle as The elbow method
        plt.title('the elbow method')
# label for x- axis
        plt.xlabel('Number of Clusters')
# label for y - axis
        plt.ylabel('Wcss')
# to show the graph
        plt.show()

#calling the elbow method to show elbow plot based on original dataset
elb_met(dataset)

from sklearn import metrics
# Define a method to get Kmean from passed dataframe and also get Silhouette score based on that.
# This method returns Silhouette score and predicted K mean values
def sil_met(sldf):
# from elbow graph we derived that optimal number of clusters for this dataset = 3
        nclusters =3
        seed = 0
# calculate Kmeans for given dataset
        km = KMeans(n_clusters=nclusters, random_state=seed)
        km.fit(sldf)
# predict the cluster for each data point
        y_cluster_km = km.predict(sldf)
#
# Calculate the Silhouette Score.
# +1: Means clusters are well apart from each other and clearly distinguished.
#  0: Means clusters are indifferent, or we can say that the distance between clusters is not significant.
# -1: Means clusters are assigned in the wrong way.
#
        score = metrics.silhouette_score(sldf,y_cluster_km)
        return(score, y_cluster_km)

# Call method to get Silhouette score for keamns on original dataset
sil_scr, y_cluster_km = sil_met(dataset)
print("Silhouette Score on Original data: ", sil_scr, '\n')

# Doing the Feature Scaling so that the larger features should not dominate the others in clustering, etc.
from sklearn import preprocessing
scaler = preprocessing.StandardScaler()
scaler.fit(dataset)
# Scaler.transform will transform our data
X_scaled_array = scaler.transform(dataset)
X_scaled = pd.DataFrame(X_scaled_array, columns= dataset.columns)
print('Dataframe after Scaling: ')
# printing the 2 sample from X_scaled
print(X_scaled.sample(2),'\n')

# Call the sil_met to get Kmean and Silhouette score based on Scaled data
sil_scr, y_scl_km = sil_met(X_scaled)
print("Silhouette Score after Scaling: ", sil_scr, '\n')

# Applying PCA and Kmean on orignal dataset
# Principal Component Analysis, or PCA, is a dimensionality-reduction method
pca = PCA(2)
X_pca = pca.fit_transform(dataset)
pcadf = pd.DataFrame(data = X_pca, columns= ['PC1', 'PC2'])
print('PCA data sample: ')
print(pcadf.sample(),'\n')

# Calling sil_met method by passing PCA dataframe and doing Kmean on it.
# method will return Silhouette Score
# *** (PCA+KMEANS) ***

pca_score, y_pca_km = sil_met(pcadf)
print("PCA Silhouette Score for original data: ", pca_score, '\n')
#print('kmean: ', y_cluster_km)

# Applying PCA and Kmean on Scaled dataset
# *** (SCALING+PCA+KMEANS) ***
X_pca = pca.fit_transform(X_scaled)
pcadf = pd.DataFrame(data = X_pca, columns= ['PC1', 'PC2'])
print(pcadf.sample(),'\n')
# Calling sil_met method by passing PCA dataframe and doing Kmean on it.
pca_score, y_scl_km = sil_met(pcadf)
print("PCA Silhouette Score after Scaling: ", pca_score,'\n')

# Visualization
# Create Dataframe with Scaled PCA data columns plus PCA based Kmeans values columns
km_df = pd.DataFrame(data = y_scl_km, columns=['Kmeans'])
pcadf2 = pd.concat([pcadf, km_df],axis = 1)
print(pcadf2.sample())

# Scatter plot for Scaled PCA data and Kmean values
sns.FacetGrid(pcadf2, hue="Kmeans", height=5).map(plt.scatter, "PC1", "PC2").add_legend()
plt.show()