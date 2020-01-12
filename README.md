# Data Mining Project

Project developed for Data Mining discipline - UFF / 2018.
Developed by: Jacó Julio.

## Project description

This project aims to analyze 3 datasets from the UCI Dataset repository. It has to be a classification, one about regression and one about clustering  analysis. You should talk about the evaluation metrics and show the best results. The preprocessing file indicates the preprocessing done in the dataset and the models file indicates the built models for the dataset. Only clustering has only the models file.

## Chosen dataset details

### Classification

The dataset chosen for classification was [Breast Cancer Wisconsin (Original)](https://archive.ics.uci.edu/ml/datasets/Breast+Cancer+Wisconsin+%28Original%29).  
Detail about dataset:

* Number of Instances: 699
* Number of Attributes: 10
* Missing Values: Yes
* Attribute Characteristics: Integer
* Attribute Details:
  1. Sample code number: id number
  2. Clump Thickness: 1 - 10
  3. Uniformity of Cell Size: 1 - 10
  4. Uniformity of Cell Shape: 1 - 10
  5. Marginal Adhesion: 1 - 10
  6. Single Epithelial Cell Size: 1 - 10
  7. Bare Nuclei: 1 - 10
  8. Bland Chromatin: 1 - 10
  9. Normal Nucleoli: 1 - 10
  10. Mitoses: 1 - 10
  11. Class: (2 for benign, 4 for malignant)

### Regression

The dataset chosen for Regression was [Concrete Compressive Strength](https://archive.ics.uci.edu/ml/datasets/Concrete+Compressive+Strength).  
Detail about dataset:

* Number of Instances: 1030
* Number of Attributes: 9
* Missing Values: No
* Attribute Characteristics: Real
* Attribute Details:
  1. Cement (component 1) -- quantitative -- kg in a m3 mixture -- Input Variable
  2. Blast Furnace Slag (component 2) -- quantitative -- kg in a m3 mixture -- Input Variable
  3. Fly Ash (component 3) -- quantitative -- kg in a m3 mixture -- Input Variable
  4. Water (component 4) -- quantitative -- kg in a m3 mixture -- Input Variable
  5. Superplasticizer (component 5) -- quantitative -- kg in a m3 mixture -- Input Variable
  6. Coarse Aggregate (component 6) -- quantitative -- kg in a m3 mixture -- Input Variable
  7. Fine Aggregate (component 7) -- quantitative -- kg in a m3 mixture -- Input Variable
  8. Age -- quantitative -- Day (1~365) -- Input Variable
  9. Concrete compressive strength -- quantitative -- MPa -- Output Variable

### Clustering

The dataset chosen for Clustering was [Seeds](https://archive.ics.uci.edu/ml/datasets/seeds).  
Detail about dataset:

* Number of Instances: 210
* Number of Attributes: 7
* Missing Values: No
* Attribute Characteristics: Real
* Attribute Details. All of these parameters were real-valued continuous:
  1. area A,
  2. perimeter P,
  3. compactness C = 4*pi*A/P^2,
  4. length of kernel,
  5. width of kernel,
  6. asymmetry coefficient
  7. length of kernel groove.


## Exploratory analysis

### Classification

Exploratory analysis and creation of the best classification model will be described step by step.

1. The class was plotted and the imbalance noticed.

![Imbalance Class](https://github.com/Jaco-Julio/DataMiningProject/blob/master/Classification/imbalance_class.png "Imbalance Class")

2. A balance was made with the SMOTE algorithm.
3. The dataset was divided into training set and test set with the train_test_split algorithm.
4. The best parameters were searched with the GridSearchCV algorithm. 
5. This search was done using Logistic Regression, KNN, SVM, Bagging Classifier, Random Forest and XGBoosting algorithms.
6. The best classification algorithms were chosen, such as Logistic Regression, Bagging Classifier, Random Forest and XGBoosting. Key assessment metrics were plotted.

### Regression

Exploratory analysis and creation of the best Regression model will be described step by step.

1. The data were plotted and outliers were observed.
![Class Distribution](https://github.com/Jaco-Julio/DataMiningProject/blob/master/Regression/distribution.png "Class Distribution")
![Class Boxplot](https://github.com/Jaco-Julio/DataMiningProject/blob/master/Regression/boxplot.png "Class Boxplot")
2. The outliers were excluded from an upper and lower limit.
![Class Distribution Without Outlier](https://github.com/Jaco-Julio/DataMiningProject/blob/master/Regression/distribution1.png "Class Distribution Without Outlier")
![Class Boxplot Without Outlier](https://github.com/Jaco-Julio/DataMiningProject/blob/master/Regression/boxplot1.png "Class Boxplot Without Outlier")
3. The data were normalized and standardized.
4. The dataset was divided into training set and test set with the train_test_split algorithm.
5. The regression was performed using the Linear Regression, Ridge Regression, Lasso Regression, ElasticNet Regression, KNN, Decision Tree, and SVM algorithms.
6. The mean absolute error, mean squared error and r2 metrics were chosen to evaluate the best model.

### Clustering

Exploratory analysis and creation of the best Clustering model will be described step by step.

1. The dataset chosen is classification and clustering at the same time, because it has the class yet.
2. By having the class, data was visualized to see its behavior. It turned out that the dataset was balanced.
![Class Distribution](https://github.com/Jaco-Julio/DataMiningProject/blob/master/Clustering/distribution.png "Class Distribution")
3. The class column was removed.
4. The data were normalized.
5. The Kmeans algorithm was clustered for the following cluster sequences, from 2 to 7. Each sequence was evaluated with the silhouette_score algorithm.
6. It was clustered with the AffinityPropagation algorithm and evaluated with the silhouette_score algorithm.
7. The AgglomerativeClustering algorithm was clustered for the following cluster sequences, from 2 to 7. Each sequence was evaluated with the silhouette_score algorithm.

## Results Analysis

### Classification

The rating models had a good result and the best one was the XGBoosting. The ROC and Confusion Matrix metrics indicated an optimal result for XGBoodting, especially the Confusion Matrix indicated more false positives than false negatives. This is good considering that the dataset is about cancer incidence analysis. False positives would result in more tests to indicate the best treatment and subsequent tests could result in cancer negative. Already, false negatives would end up releasing the patient, and this really having the disease.

| Algorithm | ROC |
| :-------------: | :--------: |
| Logistic Regression | 95,90% |
| Bagging Classifier | 96,00% |
| Random Forest | 97,11% |
| Gradient Boosting | 97,53% |

Confusion Matrix of Logistic Regression:

| ----------- | Benign Cancer | Malignant Cancer |
| :----------: | :---------: | :--------: |
| **Benign Cancer** | 141 | 3 |
| **Malignant Cancer** | 8 | 123 |

Confusion Matrix of Bagging Classifier:

| ----------- | Benign Cancer | Malignant Cancer |
| :----------: | :---------: | :--------: |
| **Benign Cancer** | 138 | 6 |
| **Malignant Cancer** | 5 | 126 |

Confusion Matrix of Random Forest: 

| ----------- | Benign Cancer | Malignant Cancer |
| :----------: | :---------: | :--------: |
| **Benign Cancer** | 139 | 5 |
| **Malignant Cancer** | 3 | 128 |

Confusion Matrix of Gradient Boosting:

| ----------- | Benign Cancer | Malignant Cancer |
| :----------: | :---------: | :--------: |
| **Benign Cancer** | 138 | 6 |
| **Malignant Cancer** | 1 | 130 |




### Regression

The best result for the regression model was with Decision Tree algorithm. This algorithm obtained the best result in the 3 training and test sets (one normalized data set, one standardized data set and one original data set) for the mean absolute error, mean squared error and R2 metrics. Interestingly the best result was in the original dataset, as the dataset is regression, it was expected that the best result would be in the normalized dataset.

#### Original Dataset

| Algorithm | MAE | MSE | R2 | 
| :-------------: | :--------: | :-------: | :-------: |
| Linear Regression | -7,57 | -89,13 | 0,44 |
| Ridge Regression | -7,57 | -89,13 | 0,44 |
| Lasso Regression | -7,58 | -89,20 | 0,44 |
| ElasticNet Regression | -7,57 | -89,17 | 0,44 |
| KNN Regression | -6,55 | -69,97 | 0,56 |
| Decision Tree Regression | -4,40 | -41,13 | 0,74 |
| Suport Vector Regression | -10,28 | -159,57 | 0,015 |

#### Normalized Dataset

| Algorithm | MAE | MSE | R2 | 
| :-------------: | :--------: | :-------: | :-------: |
| Linear Regression | -0,1443 | -0,0323 | 0,44 |
| Ridge Regression | -0,1456 | -0,0323 | 0,44 |
| Lasso Regression | -0,2005 | -0,0592 | 0,0066 |
| ElasticNet Regression | -0,2005 | -0,0592 | 0,0066 |
| KNN Regression | -0,1231 | -0,0254 | 0,56 |
| Decision Tree Regression | -0,0846 | -0,0154 | 0,73 |
| Suport Vector Regression | -0,1265 | -0,0249 | 0,57 |

#### Standardized Dataset

| Algorithm | MAE | MSE | R2 | 
| :-------------: | :--------: | :-------: | :-------: |
| Linear Regression | -0,5869 | -0,5355 | 0,44 |
| Ridge Regression | -0,5873 | -0,05355 | 0,44 |
| Lasso Regression | -0,8154 | -0,9798 | 0,0066 |
| ElasticNet Regression | -0,8154 | -0,9798 | 0,0066 |
| KNN Regression | -0,4937 | -0,4026 | 0,58 |
| Decision Tree Regression | -0,3426 | -0,2588 | 0,73 |
| Suport Vector Regression | -0,3797 | -0,2615 | 0,72 |

### Clustering

The clustering was done with 3 algorithms (K Means, Affinity Propagation and AgglomerativeClustering) and evaluated with the silhouette score, a metric that measures how similar the object is to its cluster. As is well known, the dataset has 3 labels, but in clustering two algorithms (Kmeans and AgglomerativeClustering) indicated that the dataset would be better divided into 2 labels according to the characteristics of the objects. Only one algorithm (AffinityPropagation) indicated that the dataset is best divided into 3 labels.

| Algorithm | Clusters | Silhouette Score |
| :-------: | :------: | :--------------: |
| Kmeans | 2 | 0,5051 |
| Affinity Propagation | 3 | 0,620 |
| Agglomerative Clustering | 2 | 0,4934 |




