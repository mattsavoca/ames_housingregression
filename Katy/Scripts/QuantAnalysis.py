# -*- coding: utf-8 -*-
"""
Quant Analysis
"""

# Basic Libraries
import pandas as pd
import numpy as np

# Pull in data
test = pd.read_csv('./test.csv')
train = pd.read_csv('./train.csv')
df = train.copy()
df['logy'] = np.log(df['SalePrice'])

# function to do basic EDA
def eda(colname):
    print("Mean: " + str(round(df[colname].mean(), 2)))
    print("STD: " + str(round(df[colname].std(), 3)))
    print("Range: " + str(df[colname].min()) + " to " + str(df[colname].max()))
    print("Corr to price: " + str(round(np.corrcoef(df[colname], df['SalePrice'], )[0][1],5)))
    print("Outlier count: " + str(len(detect_outlier(df[colname]))))
    #df[colname].hist()
    df.plot(colname, 'SalePrice', kind = "scatter")
      
# function to detect outliers
def detect_outlier(data_1):
    outliers=[]
    
    threshold=3
    mean_1 = np.mean(data_1)
    std_1 =np.std(data_1)
    
    for y in data_1:
        z_score= (y - mean_1)/std_1 
        if np.abs(z_score) > threshold:
            outliers.append(y)
    return outliers

# function to create QUANTDF 
def quantdf(df, y, tot):
    
    df = df._get_numeric_data()
    
    dfdf = pd.DataFrame(columns = ["range", "mean", "std", 
                                   "NAs", "non0",
                                   "outliers", "corr", "LRfit"])
    for col in df:
        
        xrange = str(df[col].min()) + " to " + str(df[col].max())
        xmean = df[col].mean()
        xstd = df[col].std()
        
        xnas = df[col].isnull().sum()
        xnon0 = sum(df[col] != 0)
        
        xoutliers = len(detect_outlier(df[col]))
              
        if tot == "train":
            from sklearn import linear_model
            x = df[col].fillna(0).values.reshape(-1,1) # fillNA for LR only
            lr = linear_model.LinearRegression()
            lr.fit(x, y)
            xfit = round(lr.score(x,y),4)  
            xcorr = round(df[col].corr(y),5)
        else:
            xfit = "na/test"
            xcorr = "na/test"
    
        dfdf.loc[col] = [xrange, xmean, xstd,
                xnas, xnon0, 
                xoutliers, xcorr, xfit]
    
    return dfdf

# Create QUANTDFs
#train_qdf = quantdf(train._get_numeric_data(), train['SalePrice'], "train")
#train_logy_qdf = quantdf(train._get_numeric_data(), np.log(train['SalePrice']), "train")
#test_qdf =  quantdf(test._get_numeric_data(), 'y', "test")

# Compare QUANTDFs
#ttqf = pd.concat([train_qdf, test_qdf], axis = 1, keys = ['train', 'test']).swaplevel(axis='columns')[train_qdf.columns[:6]]

# compare y/loy catdfs - logy is better estimator for most
#import matplotlib.pyplot as plt
#plt.plot(train_logy_qdf['corr'] - train_qdf['corr'])
#plt.plot(train_logy_qdf['LRfit'] - train_qdf['LRfit'])
#plt.axhline(y=0, color='r', linestyle='-')
#plt.show()

# Top 10 Q columns to use
#top10Qcol = list(train_qdf.sort_values('corr', ascending = False)[:10].index.values)

