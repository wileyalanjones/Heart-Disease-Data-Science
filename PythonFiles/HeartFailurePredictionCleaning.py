#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr  2 20:38:39 2025

@author: wileyjones
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

filename = '/Users/wileyjones/Desktop/CS432/Datasets/heart.csv'

hd = pd.read_csv(filename)
print(hd)

###### CLEANING #######

print(hd.dtypes)

columns = hd.columns


### Data Types ###
for col in columns:
    if str(hd[col].dtype) == "object" or col in ['FastingBS', 'HeartDisease']:
        hd[col] = hd[col].astype("category")
        
print(hd.dtypes)

### Missing Values ###

for col in columns:
    print(f'{col}: missing values {hd[col].isna().sum()}')
    
### Value Counts ###
for col in columns:
    count = hd[col].value_counts()
    print(f'{count} \n')
    
for col in columns:
    if str(hd[col].dtype) in ["float64", 'int64']:
        ax = sns.boxplot(hd, x = col)
        ax.set_title(f'{col} Boxplot')
        plt.show()
        
hd = hd[hd['RestingBP'] > 50]
hd = hd[hd['Cholesterol'] > 50]

print(hd)

hd.to_csv("heart_failure_clean.csv")
