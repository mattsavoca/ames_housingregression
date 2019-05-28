#!/usr/bin/env python
# coding: utf-8
# %%

# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import sklearn as skl
import statsmodels.api as sm
import re
from scipy.stats import norm

def explore_variables(target_name, dt):
    for col in dt.drop(target_name, 1).columns:
        if dt.dtypes[dt.columns.get_loc(col)] == 'O': # categorical variable
            f, ax = plt.subplots()
            grouped = dt.groupby(col)
            users_sorted_average = pd.DataFrame(
                {col:vals.log_price for col,vals in grouped}).mean()\
            .sort_values(ascending=False)
            ax = sns.swarmplot(
                x=col, y=target_name, data=dt, color=".25", alpha=0.2, \
                order=users_sorted_average.index)
            fig = sns.boxplot(
                x=col, y=target_name, data=dt, \
                order = users_sorted_average.index)
            ax.set_xticklabels(ax.get_xticklabels(), rotation=45)
        else: # numerical variable
            fig, ax = plt.subplots()
            ax.scatter(x=dt[col], y=dt[target_name])
            plt.ylabel(target_name, fontsize=13)
            plt.xlabel(col, fontsize=13)
            plt.show()
            
def get_nulls(df):
    nulls = pd.DataFrame(df.isnull().sum().sort_values(ascending=False))[df.isnull().sum().sort_values() > 0]
    nulls.columns = ['Null Count']
    nulls['Null Pct'] = round(nulls['Null Count']/df.shape[0],4)
    nulls.index.name = 'Feature'
    return(nulls)

def get_null_colct(df):
    nulls = pd.DataFrame(df.isnull().sum().sort_values(ascending=False))[df.isnull().sum().sort_values() > 0]
    nulls.columns = ['Null Count']
    nulls['Null Pct'] = round(nulls['Null Count']/df.shape[0],4)
    nulls.index.name = 'Feature'
    return(len(nulls))

def get_null_colnames(df):
    nulls = pd.DataFrame(df.isnull().sum().sort_values(ascending=False))[df.isnull().sum().sort_values() > 0]
    nulls.columns = ['Null Count']
    nulls['Null Pct'] = round(nulls['Null Count']/df.shape[0],4)
    nulls.index.name = 'Feature'
    return(list(nulls.index))


# %%




