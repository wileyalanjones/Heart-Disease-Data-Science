#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 12 09:52:46 2025

@author: wileyjones
"""

##### LIBRARIES ###########################################################
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.model_selection import train_test_split, cross_val_score
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn import tree
from sklearn.tree import DecisionTreeClassifier 
from sklearn.metrics import ConfusionMatrixDisplay
from imblearn.under_sampling import RandomUnderSampler
from sklearn.metrics import classification_report, roc_auc_score

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

##### DECISION TREES ######################################################

DT = DecisionTreeClassifier(max_depth=8)
DT_Model = DT.fit(train, trainLabels)
path = DT_Model.cost_complexity_pruning_path(train, trainLabels)
ccp_alphas = path.ccp_alphas

trees = []
for alpha in ccp_alphas:
    alphaTree = DecisionTreeClassifier(random_state=42, ccp_alpha=alpha)
    alphaTree.fit(train, trainLabels)
    trees.append(alphaTree)

# Evaluate using cross-validation
scores = [np.mean(cross_val_score(alphaTree, train, trainLabels, cv=5)) 
          for alphaTree in trees]

# Plot alpha vs score
plt.figure(figsize=(10, 6))
plt.plot(ccp_alphas, scores, marker='o', drawstyle="steps-post")
plt.xlabel("ccp_alpha")
plt.ylabel("Cross-validated accuracy")
plt.title("Finding Optimal Alpha for Pruning")
plt.grid()
plt.show()

# Choose best alpha (with highest score)
best_alpha = ccp_alphas[np.argmax(scores)]
print("Best ccp_alpha:", best_alpha)

DT = DecisionTreeClassifier(random_state=31, class_weight='balanced')
DT_Model = DT.fit(train, trainLabels)

### Names for chart
features = train.columns.values
classNames = DT_Model.classes_

classNames = [str(x) for x in classNames]

plt.figure(figsize=(23,10))
plot = tree.plot_tree(DT_Model,
                 feature_names=features,
                 class_names=classNames,
                 filled=True,
                 fontsize=10)
plt.show()

#### PREDICTIONS
predictions = DT_Model.predict(test)

con_mat = confusion_matrix(testLabels, predictions)
print(con_mat)

##Create the fancy CM using Seaborn
sns.heatmap(con_mat, annot=True, cmap='Purples',cbar=False, fmt='d')
plt.title("Quant CM For Decision Tree No Pruning",fontsize=20)
plt.xlabel("Actual", fontsize=15)
plt.ylabel("Predicted", fontsize=15)
plt.show()

## Confusion Matrix Plot Option 2
CM_disp=ConfusionMatrixDisplay(confusion_matrix=con_mat, display_labels=classNames)
CM_disp.plot(colorbar=False)
plt.title("Quantitative Decision Tree Confusion Matrix")
plt.show()

print(classification_report(testLabels, predictions))
print("ROC AUC:", roc_auc_score(testLabels, DT_Model.predict_proba(test)[:,1]))


##### CATEGORICAL #########################################################

QualDF = reDF.select_dtypes(include='object')

## Ordinal Encoder
mapper = {'Low': 1, "Medium": 2, "High": 3}
QualDF["Exercise Habits"] = QualDF["Exercise Habits"].replace(mapper)
QualDF['Alcohol Consumption'] = QualDF['Alcohol Consumption'].replace(mapper)
QualDF['Stress Level'] = QualDF['Stress Level'].replace(mapper)
QualDF['Sugar Consumption'] = QualDF['Sugar Consumption'].replace(mapper)

## One Hot Encoding
QualDF = pd.get_dummies(QualDF, columns=['Gender', 'Smoking', 'Family Heart Disease',
                                 'Diabetes', 'High Blood Pressure',
                                 'Low HDL Cholesterol', 'High LDL Cholesterol'],
                    drop_first=True, dtype=int)

QualDF = QualDF.drop(columns=["Gender_Male"])
QualDF

### Train Test Split
train_X, test_X = train_test_split(QualDF, test_size=.2, random_state=31)

## Save Labels
train_Y = train_X['LABEL']
test_Y = test_X['LABEL']

## Drop Labels 
train_X = train_X.drop(columns=['LABEL'])
test_X = test_X.drop(columns=['LABEL'])

## Find Alpha
DT = DecisionTreeClassifier(max_depth=8)
DT_Model = DT.fit(train_X, train_Y)
path = DT_Model.cost_complexity_pruning_path(train_X, train_Y)
ccp_alphas = path.ccp_alphas

trees = []
for alpha in ccp_alphas:
    alphaTree = DecisionTreeClassifier(random_state=42, ccp_alpha=alpha)
    alphaTree.fit(train_X, train_Y)
    trees.append(alphaTree)

# Evaluate using cross-validation
scores = [np.mean(cross_val_score(alphaTree, train_X, train_Y, cv=5)) 
          for alphaTree in trees]

# Plot alpha vs score
plt.figure(figsize=(10, 6))
plt.plot(ccp_alphas, scores, marker='o', drawstyle="steps-post")
plt.xlabel("ccp_alpha")
plt.ylabel("Cross-validated accuracy")
plt.title("Finding Optimal Alpha for Pruning")
plt.grid()
plt.show()

# Choose best alpha (with highest score)
best_alpha = ccp_alphas[np.argmax(scores)]
print("Best ccp_alpha:", best_alpha)

### Final Decision Tree
DT = DecisionTreeClassifier(random_state=31, 
                            class_weight='balanced')
DT_Model = DT.fit(train_X, train_Y)

### Names for chart
features = train_X.columns.values
classNames = DT_Model.classes_

classNames = [str(x) for x in classNames]

plt.figure(figsize=(18,10))
plot = tree.plot_tree(DT_Model,
                 feature_names=features,
                 class_names=classNames,
                 filled=True,
                 fontsize=12)
plt.show()

#### PREDICTIONS
predictions_Qual = DT_Model.predict(test_X)

con_mat = confusion_matrix(test_Y, predictions_Qual)
print(con_mat)

##Create the fancy CM using Seaborn
sns.heatmap(con_mat, annot=True, cmap='Oranges',cbar=False, fmt='d')
plt.title("Qual CM For Decision Tree No Pruning",fontsize=20)
plt.xlabel("Actual", fontsize=15)
plt.ylabel("Predicted", fontsize=15)
plt.show()

## Confusion Matrix Plot Option 2
CM_disp=ConfusionMatrixDisplay(confusion_matrix=con_mat, display_labels=classNames)
CM_disp.plot(colorbar=False)
plt.title("Qualitative Decision Tree Confusion Matrix")
plt.show()

### SCORES 
print(classification_report(test_Y, predictions_Qual))
print("ROC AUC:", roc_auc_score(test_Y, DT_Model.predict_proba(test_X)[:,1]))

##### NAIVE BAYES ###########################################################

from sklearn.naive_bayes import CategoricalNB
from sklearn.naive_bayes import BernoulliNB
from sklearn.naive_bayes import MultinomialNB
from sklearn.naive_bayes import GaussianNB

##### GUASSIAN ##############################################################

## Instance
GNB = GaussianNB()

### Train
GNB_Model = GNB.fit(train, trainLabels)

### Predictions
GNB_Pred = GNB_Model.predict(test)
G_probs = GNB_Model.predict_proba(test).round(2)

GNB_Model.score(test, testLabels)

for i in range(20):
    print(f'Yes: {G_probs[i][0]:.2f} | No: {G_probs[i][1]:.2f}')
    
yes_max = max([G_probs[y][0] for y in range(len(G_probs))])
no_max = max([G_probs[y][1] for y in range(len(G_probs))])

print('\n')
print(f'Max Yes Probablity: {yes_max}')
print(f'Max  No Probablity: {no_max}')

### Confusion Matrix 
con_mat = confusion_matrix(testLabels, GNB_Pred)
print(con_mat)

## Confusion Matrix Plot Option 2
CM_disp=ConfusionMatrixDisplay(confusion_matrix=con_mat, display_labels=GNB_Model.classes_)
CM_disp.plot(colorbar=False)
plt.title("Guasian Naive Bayes Confusion Matrix")
plt.show()

print(classification_report(testLabels, GNB_Pred))
print("ROC AUC:", roc_auc_score(testLabels, GNB_Model.predict_proba(test)[:,1]))

##### CATEGORICAL ###########################################################

### Instance
CNB = CategoricalNB()

### Train Model
CNB_Model = CNB.fit(train_X, train_Y)

### Predictions
CNB_Pred = CNB_Model.predict(test_X)

### Probablities
print(CNB_Model.predict_proba(test_X))

### Score 
CNB_Model.score(test_X, test_Y)
C_probs = CNB_Model.predict_proba(test_X).round(2)

### Max's 
print('\n')
for i in range(20):
    print(f'Yes: {C_probs[i][0]:.2f} | No: {C_probs[i][1]:.2f}')
    
yes_max = max([C_probs[y][0] for y in range(len(C_probs))])
no_max = max([C_probs[y][1] for y in range(len(C_probs))])

print('\n')
print(f'Max Yes Probablity: {yes_max}')
print(f'Max  No Probablity: {no_max}')

### Confusion Matrix 
con_mat = confusion_matrix(test_Y, CNB_Pred)
print(con_mat)

## Confusion Matrix Plot Option 2
CM_disp=ConfusionMatrixDisplay(confusion_matrix=con_mat, display_labels=CNB_Model.classes_)
CM_disp.plot(colorbar=False)
plt.title("Categorical Naive Bayes Confusion Matrix")
plt.show()

### Scores 
print(classification_report(test_Y, CNB_Pred))
print("ROC AUC:", roc_auc_score(test_Y, CNB_Model.predict_proba(test_X)[:,1]))

###### MULTINOMAL ##########################################################

### Instance
MNB = MultinomialNB()

### Train Model
MNB_Model = MNB.fit(train_X, train_Y)

### Predictions
MNB_Pred = MNB_Model.predict(test_X)

### Probablities
print(MNB_Model.predict_proba(test_X).round(3))

### Confusion Matrix 
con_mat = confusion_matrix(test_Y, MNB_Pred)
print(con_mat)

## Confusion Matrix Plot Option 2
CM_disp=ConfusionMatrixDisplay(confusion_matrix=con_mat, display_labels=MNB_Model.classes_)
CM_disp.plot(colorbar=False)
plt.show()

### Scores 
print(classification_report(test_Y, MNB_Pred))
print("ROC AUC:", roc_auc_score(test_Y, MNB_Model.predict_proba(test_X)[:,1]))