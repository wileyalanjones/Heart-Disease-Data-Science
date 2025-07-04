#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr 13 12:07:24 2025

@author: wileyjones
"""

## Imports
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from IPython.display import clear_output
from mpl_toolkits.mplot3d import Axes3D
from sklearn.cluster import KMeans
from sklearn import datasets
from sklearn.preprocessing import StandardScaler, OrdinalEncoder, OneHotEncoder
from sklearn.metrics import silhouette_score
from sklearn.metrics import silhouette_samples
from sklearn.metrics import davies_bouldin_score, calinski_harabasz_score, adjusted_rand_score

pd.set_option('display.max_columns', None)

filename = "/Users/wileyjones/Desktop/CS432/Datasets/heart_disease_clean.csv"

df = pd.read_csv(filename, index_col=0)
print(df)
df.dtypes
##### PREPARING ##########################################################

## Saving Label
LABEL = df['Heart Disease Status']
df = df.drop(columns=['Heart Disease Status'])

df = df.select_dtypes(include='number')
print(df)

"""
## Ordinal Encoder
mapper = {'Low': 1, "Medium": 2, "High": 3}
df["Exercise Habits"] = df["Exercise Habits"].replace(mapper)
df['Alcohol Consumption'] = df['Alcohol Consumption'].replace(mapper)
df['Stress Level'] = df['Stress Level'].replace(mapper)
df['Sugar Consumption'] = df['Sugar Consumption'].replace(mapper)
print(df)

## One Hot Encoding
df = pd.get_dummies(df, columns=['Gender', 'Smoking', 'Family Heart Disease',
                                 'Diabetes', 'High Blood Pressure',
                                 'Low HDL Cholesterol', 'High LDL Cholesterol'],
                    drop_first=True, dtype=int)

df = df.drop(columns=["Gender_Male"])
"""

### Standardization
scaler = StandardScaler()
scaled_df = scaler.fit_transform(df)
scaled_df

columns = df.columns
done_df = pd.DataFrame(scaled_df, columns=columns)
print(done_df)
done_df.dtypes


### Correlation Matrix
corr_matrix = df.corr(numeric_only=True)

# Set size of visualization
plt.figure(figsize=(10, 8))

sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='coolwarm',
            square=True, linewidths=0.5, cbar_kws={'shrink': 0.5})

plt.title('Correlation Matrix Heat Map')
plt.show()


### Elbow Method ###
inertia = []
k_range = range(1, 31)

for k in k_range:
    kmeans = KMeans(n_clusters=k, init='k-means++', random_state=42)
    kmeans.fit(done_df)
    inertia.append(kmeans.inertia_)
    
plt.figure(figsize=(8, 4))
plt.plot(k_range, inertia, marker='o')
plt.title('Elbow Method for Optimal k')
plt.xlabel('Number of clusters (k)')
plt.ylabel('Inertia')
plt.grid(True)
plt.show()

### Silhoutte Method
sil_scores = []
sil_range = range(2, 11)  # Start from 2 because silhouette score is undefined for k=1

for k in [5, 10, 30]:
    kmeans = KMeans(n_clusters=k, random_state=42)
    labels = kmeans.fit_predict(df)
    
    # Compute overall silhouette score
    overall_score = silhouette_score(df, labels)
    print(f"\nOverall Silhouette Score for k={k}: {overall_score:.4f}")
    
    # Compute per-cluster silhouette scores
    silhouette_vals = silhouette_samples(done_df, labels)
    print(f"Per-cluster silhouette scores for k={k}:")
    for cluster_id in range(k):
        cluster_silhouette = silhouette_vals[labels == cluster_id].mean()
        cluster_size = np.sum(labels == cluster_id)
        print(f"Cluster {cluster_id}: Avg Silhouette = {cluster_silhouette:.4f}, Size = {cluster_size}")
    
plt.figure(figsize=(8, 4))
plt.plot(sil_range, sil_scores, marker='o')
plt.title('Silhouette Scores for Different k')
plt.xlabel('Number of clusters (k)')
plt.ylabel('Silhouette Score')
plt.grid(True)
plt.show()


###### TWO CLUSTERS #######################################################

twoClusers = KMeans(
    n_clusters=2, 
    init='k-means++', 
    n_init=10, 
    random_state=42)
twoClusers.fit(done_df)

## Add Labels to DF 
df['Clusters'] = twoClusers.labels_

## Convert Centers back to initial scale
clusters_centers_2 = scaler.inverse_transform(twoClusers.cluster_centers_)
print(clusters_centers_2)

for j in range(len(clusters_centers_2[0])):
    print(f'{columns[j]} \ncluster 1: {clusters_centers_2[0][j]}\ncluster 2: {clusters_centers_2[1][j]}\n')

fig = plt.figure(figsize=(8,6)) 
ax = fig.add_subplot(projection='3d')
scatter = ax.scatter(
    df['Fasting Blood Sugar'],
    df['Cholesterol Level'], 
    df['Age'],
    c=df['Clusters'],
    cmap='coolwarm', 
    alpha=.1
    )
ax.scatter(
    clusters_centers_2[:, 6], 
    clusters_centers_2[:, 2],
    clusters_centers_2[:, 0],
    marker='x', 
    s=75,
    c='black')

ax.set_xlabel('Fasting Blood Sugar')
ax.set_ylabel('Cholesterol Level')
ax.set_zlabel('Age', labelpad=15)
ax.set_title('Heart Disease KMeans Without PCA')
plt.tight_layout()
plt.show()

######## FIVE CLUSTERS ###################################################3

fiveClusers = KMeans(
    n_clusters=5, 
    init='k-means++', 
    n_init=10, 
    random_state=42)
fiveClusers.fit(done_df)

## Add Labels to DF 
df['Clusters'] = fiveClusers.labels_
print(df)

## Convert Centers back to initial scale
clusters_centers_5 = scaler.inverse_transform(fiveClusers.cluster_centers_)
print(clusters_centers_5)

for i in range(len(clusters_centers_5[0])):
    print(columns[i])
    for j in range(len(clusters_centers_5)):
        print(clusters_centers_5[j][i])
    print('\n')


fig = plt.figure(figsize=(8,6)) 
ax = fig.add_subplot()
scatter = ax.scatter(
    df['Fasting Blood Sugar'],
    df['Triglyceride Level'], 
    #df['BMI'],
    c=df['Clusters'], 
    alpha=.15
    )
ax.scatter(
    clusters_centers_5[:, 6], 
    clusters_centers_5[:, 5],
   #clusters_centers_5[:, 3],
    marker='x', 
    s=75,
    c='black')

ax.set_xlabel('Fasting Blood Sugar')
ax.set_ylabel('Triglyceride Level')
#ax.set_zlabel('Age', labelpad=15)
ax.set_title('Five Clusters Fasting Blood Sugar vs Triglyceride Level')
plt.tight_layout()
plt.show()


######## TEN CLUSTERS #######################################################

tenClusers = KMeans(
    n_clusters=10, 
    init='k-means++', 
    n_init=10, 
    random_state=42)
tenClusers.fit(done_df)

## Add Labels to DF 
df['Clusters'] = tenClusers.labels_
print(df)

## Convert Centers back to initial scale
cluster_centers_10 = scaler.inverse_transform(tenClusers.cluster_centers_)
print(cluster_centers_10)

for i in range(len(cluster_centers_10[0])):
    print(columns[i])
    for j in range(len(cluster_centers_10)):
        print(cluster_centers_10[j][i])
    print('\n')

fig = plt.figure(figsize=(8,6)) 
ax = fig.add_subplot(projection='3d')
scatter = ax.scatter(
    df['Age'],
    df['Fasting Blood Sugar'], 
    df['BMI'],
    c=df['Clusters'], 
    alpha=.075
    )
ax.scatter(
    cluster_centers_10[:, 0], 
    cluster_centers_10[:, 6],
    cluster_centers_10[:, 3],
    marker='x', 
    s=75,
    c='black')

ax.set_xlabel('Age')
ax.set_ylabel('Fasting Blood Sugar')
ax.set_zlabel('BMI')
ax.set_title('Ten Clusters: Age, Fasting Blood Sugar and BMI')
plt.tight_layout()
plt.show()


##### PCA ###################################################################

### 3D PCA
myPCA = PCA(n_components=3)
Result = myPCA.fit_transform(done_df)
print(Result[:, 2]) 
print(Result) ## Print the new (transformed) dataset
print(len(Result))
print("The relative eigenvalues are:", myPCA.explained_variance_ratio_)
print("The actual eigenvalues are:", myPCA.explained_variance_)
EVects = myPCA.components_
print("The eigenvectors are:\n", EVects)

### Labels to Numbers
LabelNum = list(map(lambda x: 1 if x == 'Yes' else 0, LABEL))
print(LabelNum)

### Visualizing PCA with Labels 
fig2 = plt.figure()
ax2 = fig2.add_subplot(projection='3d')

x = Result[:, 0]
y = Result[:, 1] 
z = Result[:, 2]

ax2.scatter(x, y, z, cmap="RdYlGn", edgecolor='k', s=50, c=LabelNum, facecolor=None, alpha=.1)
ax2.set_xlabel('X')
ax2.set_ylabel('Y')
ax2.set_zlabel('Z')
ax2.set_title('3D PCA')
ax2.invert_zaxis()

plt.show()

### Bar Plot of PC1 Eigenvalues 
plt.bar(
    range(len(myPCA.explained_variance_ratio_)), 
    myPCA.explained_variance_ratio_,
    alpha=0.5, align='center', label='Individual Explained Variances'
)
plt.ylabel('Explained variance ratio')
plt.xlabel('Principal component index')
plt.title("Eigenvalues: Percentage of Variance/Information")
plt.tight_layout()
plt.show()

### PCA to Dataframe
PCA_df = pd.DataFrame(Result, columns=['PC1', 'PC2', 'PC3'])
print(PCA_df)

### Elbow Method ###
inertia = []
k_range = range(1, 21)

for k in k_range:
    kmeans = KMeans(n_clusters=k, init='k-means++', random_state=42)
    kmeans.fit(PCA_df)
    inertia.append(kmeans.inertia_)
    
plt.figure(figsize=(8, 4))
plt.plot(k_range, inertia, marker='o')
plt.title('Elbow Method for Optimal k')
plt.xlabel('Number of clusters (k)')
plt.ylabel('Inertia')
plt.grid(True)
plt.show()


#### Silhouette Scores 
sil_scores = []
sil_range = range(2, 31)  # Start from 2 because silhouette score is undefined for k=1

for k in sil_range:
    kmeans = KMeans(n_clusters=k, random_state=42)
    labels = kmeans.fit_predict(PCA_df)
    score = silhouette_score(PCA_df, labels)
    sil_scores.append(score)
    
plt.figure(figsize=(8, 4))
plt.plot(sil_range, sil_scores, marker='o')
plt.title('Silhouette Scores for Different k')
plt.xlabel('Number of clusters (k)')
plt.ylabel('Silhouette Score')
plt.grid(True)
plt.show()


#### TWO CLUSTERS PCA #######################################################

PCATwoClusters = KMeans(
    n_clusters=2, 
    init='k-means++', 
    n_init=10, 
    random_state=2)
PCATwoClusters.fit(PCA_df)

## Add Labels to DF 
PCA_2_labels = PCATwoClusters.labels_
print(PCA_2_labels)

PCA_2_centers = PCATwoClusters.cluster_centers_
print(PCA_2_centers)

fig = plt.figure(figsize=(8,6)) 
ax3 = fig.add_subplot(projection='3d')
scatter = ax3.scatter(
    PCA_df['PC1'],
    PCA_df['PC2'], 
    PCA_df['PC3'],
    c=PCA_2_labels, 
    alpha=.075
    )
ax3.scatter(
    PCA_2_centers[:, 0], 
    PCA_2_centers[:, 1],
    PCA_2_centers[:, 2],
    marker='x', 
    s=75,
    c='black')

ax3.set_xlabel('PC1')
ax3.set_ylabel('PC2')
ax3.set_zlabel('PC3')
ax3.set_title('PCA Two Clusters')
plt.tight_layout()
plt.show()

# Calculate metrics
davies_boulding_2 = davies_bouldin_score(PCA_df, PCATwoClusters.labels_)
calinski_harabasz_2 = calinski_harabasz_score(PCA_df, PCATwoClusters.labels_)
adj_rand_2 = adjusted_rand_score(LABEL, PCATwoClusters.labels_)

print(f'Davies-Bouldin Index: {davies_boulding_2}')
print(f'Calinski-Harabasz Index: {calinski_harabasz_2}')
print(f'Ajusted Rand Index: {adj_rand_2}')

count_2_df = pd.DataFrame(PCA_2_labels, columns=['Cluster'])
print(count_2_df)

sns.countplot(count_2_df, x='Cluster', hue='Cluster')
plt.show()

#### FOUR CLUSTERS PCA #######################################################

PCAFourClusters = KMeans(
    n_clusters=4, 
    init='k-means++', 
    n_init=10, 
    random_state=2)
PCAFourClusters.fit(PCA_df)

## Add Labels to DF 
PCA_4_labels = PCAFourClusters.labels_
print(PCA_4_labels)

PCA_4_centers = PCAFourClusters.cluster_centers_
print(PCA_4_centers)

fig = plt.figure(figsize=(8,6)) 
ax3 = fig.add_subplot(projection='3d')
scatter = ax3.scatter(
    PCA_df['PC1'],
    PCA_df['PC2'], 
    PCA_df['PC3'],
    c=PCA_4_labels, 
    alpha=.075
    )
ax3.scatter(
    PCA_4_centers[:, 0], 
    PCA_4_centers[:, 1],
    PCA_4_centers[:, 2],
    marker='x', 
    s=75,
    c='black')

ax3.set_xlabel('PC1')
ax3.set_ylabel('PC2')
ax3.set_zlabel('PC3')
ax3.set_title('PCA Four Clusters')
plt.tight_layout()
plt.show()

# Calculate metrics
davies_boulding_4 = davies_bouldin_score(PCA_df, PCAFourClusters.labels_)
calinski_harabasz_4 = calinski_harabasz_score(PCA_df, PCAFourClusters.labels_)
adj_rand_4 = adjusted_rand_score(LABEL, PCAFourClusters.labels_)

print(f'Davies-Bouldin Index: {davies_boulding_4}')
print(f'Calinski-Harabasz Index: {calinski_harabasz_4}')
print(f'Ajusted Rand Index: {adj_rand_4}')

count_4_df = pd.DataFrame(PCA_4_labels, columns=['Cluster'])
print(count_4_df)

sns.countplot(count_4_df, x='Cluster', hue='Cluster')
plt.show()

#### SIX CLUSTERS PCA #######################################################

PCASixClusters = KMeans(
    n_clusters=6, 
    init='k-means++', 
    n_init=10, 
    random_state=2)
PCASixClusters.fit(PCA_df)

## Add Labels to DF 
PCA_6_labels = PCASixClusters.labels_
print(PCA_6_labels)

PCA_6_centers = PCASixClusters.cluster_centers_
print(PCA_6_centers)

fig = plt.figure(figsize=(8,6)) 
ax3 = fig.add_subplot(projection='3d')
scatter = ax3.scatter(
    PCA_df['PC1'],
    PCA_df['PC2'], 
    PCA_df['PC3'],
    c=PCA_6_labels, 
    alpha=.05
    )
ax3.scatter(
    PCA_6_centers[:, 0], 
    PCA_6_centers[:, 1],
    PCA_6_centers[:, 2],
    marker='x', 
    s=75,
    c='black')

ax3.set_xlabel('PC1')
ax3.set_ylabel('PC2')
ax3.set_zlabel('PC3')
ax3.set_title('PCA Six Clusters')
plt.tight_layout()
plt.show()

# Calculate metrics
davies_boulding_6 = davies_bouldin_score(PCA_df, PCASixClusters.labels_)
calinski_harabasz_6 = calinski_harabasz_score(PCA_df, PCASixClusters.labels_)
adj_rand_6 = adjusted_rand_score(LABEL, PCASixClusters.labels_)

print(f'Davies-Bouldin Index: {davies_boulding_6}')
print(f'Calinski-Harabasz Index: {calinski_harabasz_6}')
print(f'Ajusted Rand Index: {adj_rand_6}')

count_6_df = pd.DataFrame(PCA_6_labels, columns=['Cluster'])
print(count_6_df)

sns.countplot(count_6_df, x='Cluster', hue='Cluster')
plt.show()


