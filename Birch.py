import os
import numpy as np 
import cv2 as cv
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from skimage.feature import hog
from sklearn.manifold import TSNE
import plotly.express as px
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.model_selection import KFold
from sklearn.cluster import Birch
import seaborn as sns
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots




# Process dataset
images = []
labels = []
for i, subfolder in enumerate(['rock', 'paper', 'scissors']):
        folder = f'./dataset/{subfolder}'
        for filename in os.listdir(folder):
            img_path = f'{folder}/{filename}'
            img = cv.imread(img_path, cv.IMREAD_GRAYSCALE)
            images.append(img)
            labels.append(i)
            
images = np.array(images)
labels = np.array(labels)




# Split dataset
x_train, x_test, y_train, y_test = train_test_split(images, labels, test_size=0.2, stratify=labels)

print(x_train.shape)
print(x_test.shape)




# HOG features
def hog_features(dataset):
    all_features = []
    for image in dataset:
        features = hog(image, pixels_per_cell=(32,32), cells_per_block=(4,4), orientations=10, block_norm='L2', visualize=False)
        all_features.append(features)
    return np.array(all_features)

x_train_hog = hog_features(x_train)
x_test_hog = hog_features(x_test)

print(x_train_hog.shape)




x_train_hog_tsne = TSNE(n_components=3, perplexity=30).fit_transform(x_train_hog)

fig = px.scatter_3d(x_train_hog_tsne, x=0, y=1, z=2, color=y_train.astype(str), title="Original clusters of train dataset using HOG features")
fig.update_layout(width=1000, height=800)  



# Sobel features
def sobel_features(dataset):
    all_features = []
    for image in dataset:
        img = cv.GaussianBlur(image, (5,5), 0) 
    
        sobelx = cv.Sobel(img, cv.CV_64F, 1, 0, ksize=7)  
        sobely = cv.Sobel(img, cv.CV_64F, 0, 1, ksize=7) 
        magnitude = np.sqrt(sobelx**2 + sobely**2)  
        
        all_features.append(magnitude.flatten())
    return np.array(all_features)
 
x_train_sobel = sobel_features(x_train)
x_test_sobel = sobel_features(x_test)

print(x_train_sobel.shape)




x_train_sobel_tsne = TSNE(n_components=3, perplexity=10).fit_transform(x_train_sobel)

fig = px.scatter_3d(x_train_sobel_tsne, x=0, y=1, z=2, color=y_train.astype(str), title="Original clusters of train dataset using Sobel features")
fig.update_layout(width=1200, height=800)  




# Birch
param_grid = {
    'n_clusters': [3],      
    'threshold': [0.1, 0.5, 1.0, 2.0],  
    'branching_factor': [10, 20, 50]   
}

kf = KFold(n_splits=5)

def tune_birch(x_train, param_grid, kf):
    results = []

    for n_clusters in param_grid['n_clusters']:
        for threshold in param_grid['threshold']:
            for branching_factor in param_grid['branching_factor']:
                fold_scores = []

                for train_idx, test_idx in kf.split(x_train):
                    x_train_fold, x_test_fold = x_train[train_idx], x_train[test_idx]

                    birch = Birch(
                        n_clusters=n_clusters,
                        threshold=threshold,
                        branching_factor=branching_factor
                    )
                    birch.fit(x_train_fold)

                    labels = birch.labels_

                    unique_labels = set(labels)
                    if 2 <= len(unique_labels) < len(x_train_fold):
                        score = silhouette_score(x_train_fold, labels)
                    else:
                        score = -1  

                    fold_scores.append(score)

                avg_score = np.mean(fold_scores)

                result = {
                    'n_clusters': n_clusters,
                    'threshold': threshold,
                    'branching_factor': branching_factor,
                    'silhouette_score': avg_score
                }
                results.append(result)

                print(f"n_clusters: {n_clusters}, threshold: {threshold}, "
                      f"branching_factor: {branching_factor}, silhouette_score: {avg_score}")

    best_result = max(results, key=lambda x: x['silhouette_score'])

    print("Best parameters:", best_result)

    return results





# Train using HOG features with PCA
hog_pca = PCA(0.90)
x_train_hog_pca = hog_pca.fit_transform(x_train_hog)
print("Number of components:", hog_pca.n_components_)

birch_hog_results = tune_birch(x_train_hog_pca, param_grid, kf)

best_params = max(birch_hog_results, key=lambda x: x['silhouette_score'])
print(best_params)



best_model_hog = Birch(
    n_clusters=best_params['n_clusters'],
    threshold=best_params['threshold'],
    branching_factor=best_params['branching_factor']
)

x_test_hog_pca = hog_pca.transform(x_test_hog)
y_pred_hog = best_model_hog.fit_predict(x_test_hog_pca)




labels, cluster_sizes = np.unique(y_pred_hog, return_counts=True)
print(f"Clusters: {labels}")
print(f"Sizes: {cluster_sizes}")




color_map = {
    0: 'red',
    1: 'green',
    2: 'blue'
}


y_test_colors = np.array([color_map[label] for label in y_test])
y_pred_hog_colors = np.array([color_map[label] for label in y_pred_hog])

x_test_hog_tsne = TSNE(n_components=3, perplexity=10).fit_transform(x_test_hog)

fig = make_subplots(rows=1, cols=2, subplot_titles=["Original Test Clusters", "Predicted Clusters"], specs=[[{"type": "scatter3d"}, {"type": "scatter3d"}]])

fig.add_trace(go.Scatter3d(x=x_test_hog_tsne[:, 0], y=x_test_hog_tsne[:, 1], z=x_test_hog_tsne[:, 2], mode='markers', marker=dict(color=y_test_colors.astype(str))), row=1, col=1)
fig.add_trace(go.Scatter3d(x=x_test_hog_tsne[:, 0], y=x_test_hog_tsne[:, 1], z=x_test_hog_tsne[:, 2], mode='markers', marker=dict(color=y_pred_hog_colors.astype(str))), row=1, col=2)

fig.update_layout(width=1200, height=600)




# Train using Sobel features with PCA
sobel_pca = PCA(0.90)
x_train_sobel_pca = sobel_pca.fit_transform(x_train_sobel)
print("Number of components:", sobel_pca.n_components_)

birch_sobel_results = tune_birch(x_train_sobel_pca, param_grid, kf)

best_params_sobel = max(birch_sobel_results, key=lambda x: x['silhouette_score'])
print(best_params_sobel)




best_model_sobel = Birch(
    n_clusters=best_params_sobel['n_clusters'],
    threshold=best_params_sobel['threshold'],
    branching_factor=best_params_sobel['branching_factor']
)

x_test_sobel_pca = sobel_pca.transform(x_test_sobel)
y_pred_sobel = best_model_sobel.fit_predict(x_test_sobel_pca)




labels_sobel, cluster_sizes_sobel = np.unique(y_pred_sobel, return_counts=True)
print(f"Clusters: {labels_sobel}")
print(f"Sizes: {cluster_sizes_sobel}")





y_pred_sobel_colors = np.array([color_map[label] for label in y_pred_sobel])

x_test_sobel_tsne = TSNE(n_components=3, perplexity=30).fit_transform(x_test_sobel)

fig = make_subplots(rows=1, cols=2, subplot_titles=["Original Test Clusters", "Predicted Clusters"], specs=[[{"type": "scatter3d"}, {"type": "scatter3d"}]])

fig.add_trace(go.Scatter3d(x=x_test_sobel_tsne[:, 0], y=x_test_sobel_tsne[:, 1], z=x_test_sobel_tsne[:, 2], mode='markers', marker=dict(color=y_test_colors.astype(str))), row=1, col=1)
fig.add_trace(go.Scatter3d(x=x_test_sobel_tsne[:, 0], y=x_test_sobel_tsne[:, 1], z=x_test_sobel_tsne[:, 2], mode='markers', marker=dict(color=y_pred_sobel_colors.astype(str))), row=1, col=2)

fig.update_layout(width=1200, height=600)




# Random clustering
random_clusters = np.random.randint(0, 3, size=len(x_train))
score_hog = silhouette_score(x_train_hog, random_clusters)
score_sobel = silhouette_score(x_train_sobel, random_clusters)

print(f"Silhouette score for HOG features: {score_hog}")
print(f"Silhouette score for Sobel features: {score_sobel}")






