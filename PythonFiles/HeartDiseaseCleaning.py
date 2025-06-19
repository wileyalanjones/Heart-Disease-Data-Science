#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr  2 19:24:21 2025

@author: wileyjones
"""

import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns

filename = "/Users/wileyjones/Desktop/CS432/Datasets/heart_disease.csv"
df = pd.read_csv(filename)

###### CLEANING ###########################################################


columns = df.columns 
print(df)

#### Missing Values ####
for col in columns:
    print(f'{col}: missing values {df[col].isna().sum()}')
    
df = df.dropna()
columns = df.columns

for col in columns:
    print(f'{col}: missing values {df[col].isna().sum()}')
    
print(df)

###### MISTAKES AND OUTLIERS ######

for col in columns:
    count = df[col].value_counts()
    print(f'{count} \n')
    
###### DATA TYPES ######

for col in columns:
    if str(df[col].dtype) == "object":
        df[col] = df[col].astype("category")

print(df.dtypes)

###### VISUALIZATIONS #######
        
for col in columns:
    if str(df[col].dtype) == "float64":
        ax = sns.boxplot(df, x = col)
        ax.set_title(f'{col} Boxplot')
        plt.show()

for col in columns:
    if str(df[col].dtype) == "float64":
        ax = sns.kdeplot(df, x = col)
        ax.set_title(f'{col} Histplot')
        plt.show()
        
for col in columns:
    if str(df[col].dtype) == "float64":
        ax = sns.kdeplot(df, 
                         x=col, 
                         hue='Heart Disease Status',
                         multiple='stack')
        ax.set_title(f'{col} Histplot')
        plt.show()

df.to_csv("heart_disease_clean.csv")

df
