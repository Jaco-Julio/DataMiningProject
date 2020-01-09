# Data Mining Project

Project developed for Data Mining discipline - UFF / 2018.
Developed by: Jacó Julio.

## Project description

This project aims to analyze 3 datasets from the UCI Dataset repository. It has to be a classification, one about regression and one about clustering  analysis. You should talk about the evaluation metrics and show the best results.

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
2. A balance was made with the SMOTE algorithm.
3. The dataset was divided into training set and test set with the train_test_split algorithm.
4. The best parameters were searched with the GridSearchCV algorithm. 
5. This search was done using Logistic Regression, KNN, SVM, Bagging Classifier, Random Forest and XGBoosting algorithms.
6. The best classification algorithms were chosen, such as Logistic Regression, Bagging Classifier, Random Forest and XGBoosting. Key assessment metrics were plotted.

### Regression

Exploratory analysis and creation of the best Regression model will be described step by step.

1. The data were plotted and outliers were observed.
2. The outliers were excluded from an upper and lower limit.
3. The data were normalized and standardized.
4. The dataset was divided into training set and test set with the train_test_split algorithm.
5. The regression was performed using the Linear Regression, Ridge Regression, Lasso Regression, ElasticNet Regression, KNN, Decision Tree, and SVM algorithms.
6. The mean absolute error, mean squared error and r2 metrics were chosen to evaluate the best model.

### Clustering

Exploratory analysis and creation of the best Clustering model will be described step by step.

1. The dataset chosen is classification and clustering at the same time, because it has the class yet.
2. By having the class, data was visualized to see its behavior. It turned out that the dataset was balanced.
3. The class column was removed.
4. The data were normalized.
5. The Kmeans algorithm was clustered for the following cluster sequences, from 2 to 7. Each sequence was evaluated with the silhouette_score algorithm.
6. It was clustered with the AffinityPropagation algorithm and evaluated with the silhouette_score algorithm.
7. The AgglomerativeClustering algorithm was clustered for the following cluster sequences, from 2 to 7. Each sequence was evaluated with the silhouette_score algorithm.






