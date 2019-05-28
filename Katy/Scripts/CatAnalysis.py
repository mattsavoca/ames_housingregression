# -*- coding: utf-8 -*-
"""
 CATS ANALYSIS
"""

# Basic Libraries
import pandas as pd
import numpy as np

# Pull in data
test = pd.read_csv('./test.csv')
train = pd.read_csv('./train.csv')
df = train.copy()
df['logy'] = np.log(df['SalePrice'])

# function for CATS    
def cda(colname): 
    print(pd.concat([df[colname].value_counts(), test[colname].value_counts()], axis = 1, sort = False))
    df.boxplot('SalePrice', colname)
    print(df.groupby(colname).mean()['SalePrice'])

### function to get metrics for CAT cols - fits 2xLR: dummy & one.v.all (one is top/mode)
def catdf(df,y,tot):
    
    df = df.drop(df._get_numeric_data().columns, axis = 1)
    
    dfdf = pd.DataFrame(columns = ["unique", "set", 
                                   "mode", "mode%", "NAs",
                                   "dummyLRscore", "ovaLRscore", 
                                   "quantLRscore", "suggest"])
    for col in df:
        
        temp = df.describe()
        quantcol = ['BsmtQual', 'BsmtCond', 'KitchenQual', 'ExterQual', 'ExterCond', 
            'GarageQual', 'GarageCond', 'HeatingQC', 'FireplaceQu', 'PoolQC', 'OverallCond', 'OverallQual']
        
        xunique = temp.loc['unique', col]
        xset = df[col].unique()
        
        xmode = temp.loc['top', col]
        xmodep = round((temp.loc['freq', col] / df.shape[0]) *100, 2)
        xnas = df.shape[0] - temp.loc['count', col]

        if tot == "train":
            from sklearn import linear_model
            xdummy = pd.get_dummies(df[col], drop_first=True)
            lrdummy = linear_model.LinearRegression()
            lrdummy.fit(xdummy, y)
            
            xova = df[col].eq(xmode).mul(1).values.reshape(-1, 1)
            lrova = linear_model.LinearRegression()
            lrova.fit(xova, y)
            
            xcorr = round(lrdummy.score(xdummy,y),4)
            xcorr2 = round(lrova.score(xova, y),4) 
            xcorr3 = 0
            
            # only if in QUANTABLE columns
            if col in quantcol:
                if col in ['OverallCond', 'OverallQual']:
                    xquant = df[col].astype(int).values.reshape(-1, 1)
                else:
                    xquant = df[col].fillna(0).replace('None', 0).replace('Po', 1).replace('Fa',2).replace('TA', 3).replace('Gd',4).replace('Ex',5).values.reshape(-1, 1)
                lrquant = linear_model.LinearRegression()
                lrquant.fit(xquant, y)
                xcorr3 = round(lrquant.score(xquant, y),4)
            
            # determine action based on metrics
            if xnas >= df.shape[0]*.9:
                xaction = "ignore"
            elif xunique == 2:
                xaction = "binary"
            elif (xcorr2 + .01) > xcorr:
                xaction = "1vA"
            elif (xcorr3 + .01) > xcorr:
                xaction = "quantify"
            else:
                xaction = "dummify"
            
        else:
            xcorr = "test"
            xcorr2 = "test"
            xcorr3 = "test"
            xaction = "test"
    
        dfdf.loc[col] = [xunique, xset, xmode, xmodep, xnas, xcorr, xcorr2, xcorr3, xaction]
        
        # save results so can duplicate on test set later
        dfdf.to_csv("FeatureSuggestion.csv", index = True)
    
    return dfdf

# Get all cats cols
#cat_df = df.drop(df._get_numeric_data().columns, axis = 1)

# Create cat df
#train_catdf = catdf(cat_df, df['SalePrice'], 'train')
#train_logy_catdf = catdf(cat_df, df['logy'], 'train')
#test_catdf = catdf(test.drop(test._get_numeric_data().columns, axis = 1), "na", 'test')

# compare test/train catdfs - tops are same, some differences in NA/diversity
#ttcatdf = pd.concat([train_catdf, test_catdf], axis = 1, keys = ['train', 'test']).swaplevel(axis='columns')[train_catdf.columns[:5]]

# compare y/loy catdfs - logy is better estimator for most
#import matplotlib.pyplot as plt
#plt.plot(train_logy_catdf.dummyLRscore - train_catdf.dummyLRscore)
#plt.plot(train_logy_catdf.ovaLRscore - train_catdf.ovaLRscore)
#plt.axhline(y=0, color='r', linestyle='-')
#plt.show()

# Top 10 CATs to use:
#top10Ccol = list(train_catdf.sort_values('dummyLRscore', ascending = False)[:10].index.values)
