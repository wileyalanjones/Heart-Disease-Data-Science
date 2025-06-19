#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 28 18:49:56 2025

@author: wileyjones
"""

import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.linear_model import RidgeCV, LassoCV
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler, OrdinalEncoder, OneHotEncoder
from mpl_toolkits.mplot3d import Axes3D
from sklearn.metrics import confusion_matrix

##### LOADING FILE #######################################################

pd.set_option('display.max_columns', None)
filename = "/Users/wileyjones/Desktop/CS432/Datasets/heart_disease_clean.csv"
df = pd.read_csv(filename, index_col=0)


##### PREPARING ##########################################################

df2 = pd.read_csv(filename, index_col=0)

## Saving Label
LABELS = df2['Heart Disease Status']
df2["Heart Disease Status"].value_counts()
df = df.drop(columns=['Heart Disease Status'])
df = df.select_dtypes(include='number')
df = df.drop(columns=['Age', "BMI", "Sleep Hours"])

###### P-VALUES ############################################################
X = df.drop(columns=["Blood Pressure"])
Y = df['Blood Pressure']
C = sm.add_constant(X)
model = sm.OLS(Y, X).fit()
print(model.summary())

### VIF 
vif_data = pd.DataFrame()
vif_data['feature'] = X.columns

vif_data["VIF"] = [variance_inflation_factor(X.values, i) 
                   for i in range(len(X.columns))]

print(vif_data)

### Correlation Matrix 
corr = X.corr()
plt.figure(figsize=(10, 8))
mask = np.triu(np.ones_like(corr, dtype=bool))
cmap = sns.diverging_palette(230, 20, as_cmap=True)
sns.heatmap(corr, annot=True, cmap='coolwarm', fmt='.2f', mask=mask)
plt.title('Feature Correlation Heatmap')
plt.show()

##### TRAIN TEST SPLIT ######################################################
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=.2,
                                                    random_state=31)

### Ridge Regression 
alphas = [0.01, 0.1, 1.0, 10.0, 100.0]  # Try a range of alpha values

ridge_cv = RidgeCV(alphas=alphas, store_cv_values=True)
ridge_cv.fit(X_train, Y_train)

print("Best alpha (Ridge):", ridge_cv.alpha_)
print("Ridge coefficients:\n", pd.Series(ridge_cv.coef_, index=X.columns))

### Lasso Regression
lasso_cv = LassoCV(alphas=[0.0001, 0.001, 0.01, 0.1, 1, 10], cv=5, random_state=42, max_iter=10000)
lasso_cv.fit(X_train, Y_train)

print("Best alpha (Lasso):", lasso_cv.alpha_)
print("Lasso coefficients:\n", pd.Series(lasso_cv.coef_, index=X.columns))

y_pred = lasso_cv.predict(X_test)
r2 = r2_score(Y_test, y_pred)
print("R² on test set:", r2)
rmse = np.sqrt(mean_squared_error(Y_test, y_pred))
print("RMSE on test set:", rmse)

y_pred_ridge = ridge_cv.predict(X_test)
r2 = r2_score(Y_test, y_pred_ridge)
print("R² on test set:", r2)
rmse = np.sqrt(mean_squared_error(Y_test, y_pred_ridge))
print("RMSE on test set:", rmse)

##### LINEAR REGRESSION ####################################################

LR = LinearRegression()
LR.fit(X_train, Y_train)
print(f'These are the coefficients: {LR.coef_}') ## This is the slope
print(f'This is the intercept: {LR.intercept_}') ## This is the y-intercept


MyPrediction = LR.predict(X_test)
## The above will predict the study hours
print(MyPrediction)
## Here, we can also look at the actual Testing data Y values 
print(Y_test)
print(LR.score(X_train,Y_train))

#### 3D 
Z = df[['CRP Level', 'Homocysteine Level']]
LR_3d = LinearRegression()
LR_3d.fit(Z, Y)

fig = plt.figure(figsize=(10,7))
ax = fig.add_subplot(projection="3d")

ax.scatter(Z['CRP Level'], Z['Homocysteine Level'], Y, color='blue',
           label="Data points", alpha=.1)

x1_range = np.linspace(Z['CRP Level'].min(), Z['CRP Level'].max(), 20)
x2_range = np.linspace(Z['Homocysteine Level'].min(), 
                       Z['Homocysteine Level'].max(),
                       20)
x1_grid, x2_grid = np.meshgrid(x1_range, x2_range)
X_grid = np.c_[x1_grid.ravel(), x2_grid.ravel()]
z_pred = LR_3d.predict(X_grid).reshape(x1_grid.shape)

# Plot the regression plane
ax.plot_surface(x1_grid, x2_grid, z_pred, color='red', alpha=0.5, label='Regression plane')

# Labels and title
ax.set_xlabel('CRP Level')
ax.set_ylabel('Homocysteine Level')
ax.set_zlabel('Blood Pressure')
plt.title('Linear Regression with Two Independent Variables')
#ax.view_init(azim=285) 
ax.view_init(elev=0)
plt.show()

############################################################################

##### LOGISTIC REGRESSION ##################################################

from imblearn.under_sampling import RandomUnderSampler
from sklearn.linear_model import LogisticRegression
import math


pd.set_option('display.max_columns', None)
filename = "/Users/wileyjones/Desktop/CS432/Datasets/heart_disease_clean.csv"
df = pd.read_csv(filename, index_col=0)
df.dtypes

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

## Save Label
df['Heart Disease Status'] = df['Heart Disease Status'].replace({"Yes": 1, "No": 0})
LABEL = df['Heart Disease Status']

## Remove unwanted columns
columns_to_keep = ['Age', 'Exercise Habits', 'Alcohol Consumption', 
                   'Stress Level', 'Sleep Hours', 'Sugar Consumption',
                   'Smoking_Yes', 'Family Heart Disease_Yes','Diabetes_Yes']

cols_delete = [col for col in df.columns if col not in columns_to_keep]
df = df.drop(columns=cols_delete)
df['LABEL'] = LABEL

##### UNDERSAMPLING #########################################################

## Sampling down the data
ax = sns.countplot(df, x="LABEL", hue="LABEL")
for container in ax.containers:
    ax.bar_label(container)
ax.set_title("Count of Label Entries in Dataset")
plt.show()

## Create Over Sampler Object
rus = RandomUnderSampler()

## Save Labels
labels = df["LABEL"].tolist()
df = df.drop(columns=["LABEL"])
print(df)

## Over Sample 
resam, re_labels = rus.fit_resample(df, labels)
resam.insert(0, "LABEL", re_labels)
print(resam)

## Show Balanced data
ax = sns.countplot(resam, x="LABEL", hue="LABEL")
for container in ax.containers:
    ax.bar_label(container)
ax.set_title("Count of Label Entries in Dataset")
plt.show()

resam = resam.drop(columns=["LABEL"])
resam
##### TEST TRAIN SPLIT #####################################################

##Split 
train, test = train_test_split(resam, test_size=.2, random_state=31)

## Save Labels
trainLabels = train['LABEL']
testLabels = test['LABEL']

## Drop Labels 
train = train.drop(columns=['LABEL'])
test = test.drop(columns=['LABEL'])

##### REGRESSION ############################################################

LogReg = LogisticRegression()
LogRegModel = LogReg.fit(train, trainLabels)

## Predictions
Predictions = LogRegModel.predict(test)

## Confustion Matrix
ConMat = confusion_matrix(testLabels, Predictions)

##Create the fancy CM using Seaborn
sns.heatmap(ConMat, annot=True, cmap='Greens',cbar=False, fmt='d')
plt.title("Confusion Matrix For Logistic Regression",fontsize=20)
plt.xlabel("Actual", fontsize=15)
plt.ylabel("Predicted", fontsize=15)
plt.show()

## Accuracy Score
print(LogRegModel.predict_proba(test))

## Equation 
print(LogRegModel.coef_.round(2))
print(LogRegModel.intercept_)

train.dtypes





























