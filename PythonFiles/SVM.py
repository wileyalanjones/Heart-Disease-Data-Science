#!/usr/bin/env python3
me # -*- coding: utf-8 -*-
"""
Created on Fri May 23 14:42:30 2025

@author: wileyjones
"""

import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np
import plotly.express as px
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.preprocessing import StandardScaler
from imblearn.under_sampling import RandomUnderSampler
from sklearn.metrics import classification_report, roc_auc_score
import seaborn as sns
from sklearn.model_selection import GridSearchCV, KFold, cross_val_score
from sklearn.ensemble import RandomForestClassifier

##### DATAFRAME ###########################################################

filename = "/Users/wileyjones/Desktop/CS432/Datasets/heart_disease_clean.csv"
df = pd.read_csv(filename, index_col=0)
pd.set_option('display.max_columns', None)

##### PREPARING DATASET ###################################################

### Undersample
rus = RandomUnderSampler()

## Save Labels
labels = df["Heart Disease Status"].tolist()
df = df.drop(columns=["Heart Disease Status"])
print(df)

## Under Sample 
reDF, RSlabels = rus.fit_resample(df, labels)
reDF.insert(0, "LABEL", RSlabels)

### Remove Non-Quantitative Columns
LABEL = reDF['LABEL']
QuantDF = reDF.select_dtypes(include='number')

### Readd Label
QuantDF['LABEL'] = LABEL
QuantDF

### Train Test Split
train, test = train_test_split(QuantDF, test_size=.2, random_state=31)

## Save Labels
trainLabels = train['LABEL']
testLabels = test['LABEL']

## Drop Labels 
train = train.drop(columns=['LABEL'])
test = test.drop(columns=['LABEL'])


##### SUPPORT VECTOR MACHINES ##############################################

### 2 POLY WITH 1 COST ##################################
Poly2 = SVC(C=1, kernel='poly', degree=2)
Poly2.fit(train, trainLabels)
Classes = Poly2.classes_

# Predictions
predictionsPoly2 = Poly2.predict(test)

# Confusion Matrix
CM = confusion_matrix(testLabels, predictionsPoly2)
disp = ConfusionMatrixDisplay(CM, display_labels=Classes)

# Heatmap
sns.heatmap(CM, annot=True, cmap='Blues', cbar=False, fmt='d')
plt.title("Confusion Matrix For Poly Degree 2 Cost=1",fontsize=20)
plt.xlabel("Predicted", fontsize=15)
plt.ylabel("Actual", fontsize=15)
plt.show()

print(classification_report(testLabels, predictionsPoly2))
print("ROC AUC:", roc_auc_score(testLabels, Poly2.predict_proba(test)[:,1]))

### RBF ##################################
RBF = SVC(C=10, coef0=-1, gamma=.01, probability=True)
RBF.fit(train, trainLabels)
Classes = RBF.classes_

# Predictions
predictionsRBF = RBF.predict(test)

# Confusion Matrix
CM = confusion_matrix(testLabels, predictionsRBF)

## Confusion Matrix Plot Option 2
disp=ConfusionMatrixDisplay(confusion_matrix=CM, display_labels=Classes)
disp.plot(colorbar=False)
plt.title("Confusion Matrix for RBF Cost=100")
plt.show()

# Heatmap
sns.heatmap(CM, annot=True, cmap='Greens', cbar=False, fmt='d')
plt.title("Confusion Matrix For RBF Cost=100",fontsize=20)
plt.xlabel("Predicted", fontsize=15)
plt.ylabel("Actual", fontsize=15)
plt.show()

print(classification_report(testLabels, predictionsRBF))
print("ROC AUC:", roc_auc_score(testLabels, RBF.predict_proba(test)[:,1]))
print(RBF.score(test, testLabels))


### Sigmoid ##################################
SIG = SVC(C=100, kernel='sigmoid', probability=True)
SIG.fit(train, trainLabels)
Classes = SIG.classes_

# Predictions
predictionsSIG = SIG.predict(test)

# Confusion Matrix
CM = confusion_matrix(testLabels, predictionsSIG)

## Confusion Matrix Plot Option 2
disp=ConfusionMatrixDisplay(confusion_matrix=CM, display_labels=Classes)
disp.plot(colorbar=False)
plt.title("Confusion Matrix for Sigmoid Cost=100")
plt.show()

# Heatmap
sns.heatmap(CM, annot=True, cmap='Reds', cbar=False, fmt='d')
plt.title("Confusion Matrix For Sigmoid Cost=100",fontsize=20)
plt.xlabel("Predicted", fontsize=15)
plt.ylabel("Actual", fontsize=15)
plt.show()

print(classification_report(testLabels, predictionsSIG))
print("ROC AUC:", roc_auc_score(testLabels, SIG.predict_proba(test)[:,1]))

kf = KFold(n_splits=5, shuffle=True, random_state=31)
scores = cross_val_score(SIG, train, trainLabels, cv=kf, scoring='accuracy')

print("Cross-validation scores:\n", scores)
print("Mean accuracy: ", scores .mean())

#### Hyper Parameter Tuning #####
param_grid = {
    'C': [1, 10, 50, 100],
    'gamma': [0.001, 0.01, 0.1, 1],
    'coef0': [-1, 0, 1, 5]
}

model = SVC()

grid_search = GridSearchCV(model, param_grid, cv=5, scoring='accuracy')
grid_search.fit(train, trainLabels)

print("Best parameters:\n", grid_search.best_params_)
print("Best cross-validated score:\n", grid_search.best_score_)

#### RANDOM FOREST #########################################################

RF = RandomForestClassifier(n_estimators=100, min_samples_split=5, min_samples_leaf=4,
                            max_features=.03, max_depth=20)
RF.fit(train, trainLabels)

RFPred = RF.predict(test)

CM = confusion_matrix(testLabels, RFPred)
disp = ConfusionMatrixDisplay(confusion_matrix=CM, display_labels=RF.classes_)
disp.plot(colorbar=False)
plt.title("CM for Random Forest Optimal Features")
plt.show()

print(RF.feature_importances_)
feature_imp = pd.Series(RF.feature_importances_, index=train.columns.values).sort_values(ascending=False)
print(feature_imp)
print(classification_report(testLabels, RFPred))


