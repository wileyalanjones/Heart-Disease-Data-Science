#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 26 18:37:39 2025

@author: wileyjones
"""

from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay
import matplotlib.pyplot as plt
from IPython.display import Image
from sklearn import tree
from sklearn.tree import plot_tree
from imblearn.under_sampling import RandomUnderSampler
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV

##### SETUP #################################################################

filename = "/Users/wileyjones/Desktop/CS432/Datasets/heart_disease_clean.csv"
df = pd.read_csv(filename, index_col=0)
pd.set_option('display.max_columns', None)

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

### Undersample
rus = RandomUnderSampler()

## Save Labels
labels = df["Heart Disease Status"].tolist()
df = df.drop(columns=["Heart Disease Status"])

## Under Sample 
reDF, RSlabels = rus.fit_resample(df, labels)
reDF.insert(0, "LABEL", RSlabels)

## Back to DF
df = reDF

### Train Test Split
train, test = train_test_split(df, test_size=.2, random_state=31)

## Save Labels
trainLabels = train['LABEL']
testLabels = test['LABEL']

## Drop Labels 
train = train.drop(columns=['LABEL'])
test = test.drop(columns=['LABEL'])

##### RANDOM FOREST ########################################################

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

###### HYPER PARAMETERS TUNING #############################################

param_dist = {
    'n_estimators': [100, 200, 300, 400, 500, 600],
    'max_depth': [5, 10, 15, 20, 25, 30, None],
    'min_samples_split': [2, 5, 10, 15, 20],
    'min_samples_leaf': [1, 2, 4, 8, 12],
    'max_features': ['sqrt', 'log2', 0.2, 0.3, 0.4, 0.5, 0.6],
    'bootstrap': [True, False]
}

# Create Random Forest
rf = RandomForestClassifier(random_state=42)

# Random search with 5-fold cross validation
random_search = RandomizedSearchCV(
    estimator=rf,
    param_distributions=param_dist,
    n_iter=100,  # Number of parameter combinations to try
    cv=5,
    scoring='accuracy',
    n_jobs=-1,
    verbose=1,
    random_state=42
)

# Fit the random search
random_search.fit(train, trainLabels)

print("Best parameters:", random_search.best_params_)
print("Best cross-validation score:", random_search.best_score_)

for i in range(3):
    next_tree = RF.estimators_[i]
    plt.figure(figsize=(50, 40))
    plot_tree(next_tree, feature_names=train.columns.values, 
          class_names=RF.classes_, filled=True, fontsize=12)
    plt.show()