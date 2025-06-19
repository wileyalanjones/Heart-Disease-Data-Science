#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 26 18:37:39 2025

@author: wileyjones
"""

from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, StackingClassifier
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
from sklearn.model_selection import GridSearchCV
from sklearn.tree import DecisionTreeClassifier 
from xgboost import XGBClassifier
from sklearn.svm import SVC 
from sklearn.linear_model import LogisticRegression

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
    
    
###### ADABOOST #############################################################

ADA = AdaBoostClassifier()
AB = ADA.fit(train, trainLabels)
AdaPred = AB.predict(test)

CM = confusion_matrix(testLabels, AdaPred)
disp = ConfusionMatrixDisplay(confusion_matrix=CM, display_labels=ADA.classes_)
disp.plot(colorbar=False)
plt.title("CM for AdaBoost Default Parameters")
plt.show()

print(classification_report(testLabels, AdaPred))

##### HYPER PARAMETER TESTING ##############################################

## Parameters to check
param_grid = {
        'n_estimators': [50, 100, 200],
        'learning_rate': [.01, .1, 1.0],
        'estimator__max_depth': [1, 2, 3]
    }

## Base of AdaBoost
base = DecisionTreeClassifier()

## Instance
adb = AdaBoostClassifier(estimator=base)

## Grid Instance 
grid = GridSearchCV(estimator=adb, param_grid=param_grid, cv=5, scoring='accuracy')
grid.fit(train, trainLabels)

## Scores 
print('\n')
print("Best Parameters:\n", grid.best_params_)
print("\nBest Score:", grid.best_score_)

###### ADABOOST OPTIMAL ####################################################

base = DecisionTreeClassifier(max_depth=3, class_weight='balanced')

ADA = AdaBoostClassifier(estimator=base, learning_rate=.1, n_estimators=100)
AB = ADA.fit(train, trainLabels)
AdaPred = AB.predict(test)

CM = confusion_matrix(testLabels, AdaPred)
disp = ConfusionMatrixDisplay(confusion_matrix=CM, display_labels=ADA.classes_)
disp.plot(colorbar=False)
plt.title("CM for AdaBoost Tuned Parameters")
plt.show()

print(classification_report(testLabels, AdaPred))

ADAProbs = ADA.predict_proba(test)

print('\nAda Boost Probabilities')
for i in range(20):
    print(f'No: {ADAProbs[i][0]:.3f} | Yes: {ADAProbs[i][1]:.3f}')

ADAProbsNo = [x[0] for x in ADAProbs]
ADAProbsYes = [y[1] for y in ADAProbs]

print("\n" , f"Max No: {max(ADAProbsNo):.3f} | Max Yes: {max(ADAProbsYes):.3f}")

##### XGBoost ##############################################################

trainLabelNums = trainLabels.apply(lambda x: 0 if x == "No" else 1)
testLabelNums = testLabels.apply(lambda x: 0 if x == "No" else 1)

param_grid = {
    'n_estimators': [100, 200],
    'max_depth': [3, 5, 7],
    'learning_rate': [0.01, 0.1],
    'subsample': [0.8, 1.0],
    'colsample_bytree': [0.8, 1.0]
}

xgb = XGBClassifier(use_label_encoder=False, eval_metric='logloss')

grid = GridSearchCV(xgb, param_grid, cv=5, scoring='f1_macro')
grid.fit(train, trainLabelNums)

print("Best Params:", grid.best_params_)

model = XGBClassifier(
    n_estimators=200,
    learning_rate=.1,
    colsample_bytree=.8,
    max_depth=7,
    subsample=1)

model.fit(train, trainLabelNums)

predictions = model.predict(test)

print(classification_report(testLabelNums, predictions))


##### STACKING #############################################################

estimators = [
        ('rf', RandomForestClassifier(n_estimators=100)),
        ('svm', SVC(kernel='sigmoid',C=10)),
        ('ada', AdaBoostClassifier())
    ]

Stacker = StackingClassifier(
        estimators=estimators, final_estimator=LogisticRegression()
    )

Stacker = Stacker.fit(train, trainLabels)
StackPred = Stacker.predict(test)

CM = confusion_matrix(testLabels, StackPred)
disp = ConfusionMatrixDisplay(confusion_matrix=CM, display_labels=Stacker.classes_)
disp.plot(colorbar=False)
plt.title("CM for Stacking Model")
plt.show()

print(classification_report(testLabels, StackPred))
print(Stacker.predict_proba(test))

StackProbs = Stacker.predict_proba(test)

print('\nStacker Probabilities')
for i in range(20):
    print(f'No: {StackProbs[i][0]:.3f} | Yes: {StackProbs[i][1]:.3f}')
    
    

