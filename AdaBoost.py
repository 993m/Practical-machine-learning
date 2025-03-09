import os
import numpy as np 
import cv2 as cv
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from skimage.feature import hog
from sklearn.manifold import TSNE
import plotly.express as px
from sklearn.ensemble import AdaBoostClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.decomposition import PCA
import seaborn as sns
import pandas as pd

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




plt.figure(figsize=(10, 5))

for i in range(5):
    for j in range(3):  
        plt.subplot(3, 5, j * 5 + i + 1)  
        plt.imshow(images[j * 800 + i], cmap='gray')
        plt.axis('off')

plt.show()




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





plt.figure(figsize=(10, 5))

for i in range(5):
    for j in range(3):  
        plt.subplot(3, 5, j * 5 + i + 1)  
        sobel_image = x_train_sobel[j * 800 + i].reshape(200, 300)
        plt.imshow(sobel_image, cmap='gray')
        plt.axis('off')

plt.show()





x_train_sobel_tsne = TSNE(n_components=3, perplexity=10).fit_transform(x_train_sobel)

fig = px.scatter_3d(x_train_sobel_tsne, x=0, y=1, z=2, color=y_train.astype(str), title="Original clusters of train dataset using Sobel features")
fig.update_layout(width=1200, height=800)  





# AdaBoost model
AdaBoost_model = AdaBoostClassifier()

param_grid = {
    'n_estimators': [50, 100, 200, 300],
    'learning_rate': [0.01, 0.1, 1.0, 10.0]
}




# Train using HOG features
hog_pca = PCA(0.95)
x_train_hog_pca = hog_pca.fit_transform(x_train_hog)
print("Number of components:", hog_pca.n_components_)

grid_search = GridSearchCV(AdaBoost_model, param_grid, cv=5, scoring='accuracy', n_jobs=-1, verbose=3)
grid_search.fit(x_train_hog_pca, y_train)
best_model_hog = grid_search.best_estimator_
print("Best parameters:", grid_search.best_params_)

# Test the best model on the test dataset
x_test_hog_pca = hog_pca.transform(x_test_hog)  
y_pred_hog = best_model_hog.predict(x_test_hog_pca)





# Results
conf_matrix_hog = confusion_matrix(y_test, y_pred_hog)
sns.heatmap(conf_matrix_hog, annot=True, fmt="d", cmap="Purples")
plt.title("Confusion matrix")
plt.xlabel("Predicted labels")
plt.ylabel("True labels")
plt.show()

results_hog = pd.DataFrame(grid_search.cv_results_)
tuning_matrix = results_hog.pivot(index="param_n_estimators", columns="param_learning_rate", values="mean_test_score")
sns.heatmap(tuning_matrix, annot=True, cmap="Purples")
plt.title("Hyperparameter tuning")
plt.show()





for param in param_grid.keys():
    df = pd.DataFrame({
        "x": results_hog[f"param_{param}"],
        "y": results_hog["mean_test_score"],
        "y_std": results_hog["std_test_score"]
    })

    df = df.groupby("x", as_index=False).mean()

    sns.lineplot(x=df["x"], y=df["y"])

    plt.fill_between(df["x"], df["y"] - df["y_std"], df["y"] + df["y_std"], color="lightblue")
    
    plt.xlabel(param)
    plt.ylabel("mean accuracy")
    plt.title(f"Performance for {param}")
    plt.grid()
    plt.show()




report_hog = classification_report(y_test, y_pred_hog)
print("Classification report hog:\n", report_hog)





# Train using Sobel features
sobel_pca = PCA(0.80) 
x_train_sobel_pca = sobel_pca.fit_transform(x_train_sobel)
print("Number of components:", sobel_pca.n_components_)

grid_search = GridSearchCV(AdaBoost_model, param_grid, cv=5, scoring='accuracy', n_jobs=-1, verbose=3)
grid_search.fit(x_train_sobel_pca, y_train)
best_model_sobel = grid_search.best_estimator_
print("Best parameters:", grid_search.best_params_)

# Test the best model on the test dataset
x_test_sobel_pca = sobel_pca.transform(x_test_sobel)  
y_pred_sobel = best_model_sobel.predict(x_test_sobel_pca)




# Results
conf_matrix_sobel = confusion_matrix(y_test, y_pred_sobel)
sns.heatmap(conf_matrix_sobel, annot=True, fmt="d", cmap="Purples")
plt.title("Confusion matrix")
plt.xlabel("Predicted labels")
plt.ylabel("True labels")
plt.show()

results_sobel = pd.DataFrame(grid_search.cv_results_)
tuning_matrix = results_sobel.pivot(index="param_n_estimators", columns="param_learning_rate", values="mean_test_score")
sns.heatmap(tuning_matrix, annot=True, cmap="Purples")
plt.title("Hyperparameter tuning")
plt.show()




for param in param_grid.keys():
    df = pd.DataFrame({
        "x": results_sobel[f"param_{param}"],
        "y": results_sobel["mean_test_score"],
        "y_std": results_sobel["std_test_score"]
    })

    df = df.groupby("x", as_index=False).mean()

    sns.lineplot(x=df["x"], y=df["y"])

    plt.fill_between(df["x"], df["y"] - df["y_std"], df["y"] + df["y_std"], color="lightblue")
    
    plt.xlabel(param)
    plt.ylabel("mean accuracy")
    plt.title(f"Performance for {param}")
    plt.grid()
    plt.show()





report_sobel = classification_report(y_test, y_pred_sobel)
print("Classification report sobel:\n", report_sobel)



