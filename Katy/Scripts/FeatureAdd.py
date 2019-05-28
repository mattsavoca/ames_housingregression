# -*- coding: utf-8 -*-
"""
Function Fun!
- beefup(df) adds new features
- nafix(df) imputes NA per specific columns
- q_to_cat(df, cols) convert quant types to string/cat
- optcat(df, y, "train"/"test") convert categories based on suggested

-thewholeshabang(df, "train"/"test")
"""

# Basic Libraries
import pandas as pd
import numpy as np

# Pull in data
#test = pd.read_csv('./test.csv')
train = pd.read_csv('./train.csv')
#df = train.copy()

#%% function adds bunch of shiny new features
def beefup(ogdf):
    
    df = ogdf.copy()
    
    df['TotalBath'] = df['FullBath'] + df['HalfBath']
    df['BsmtTotBath'] = df['BsmtHalfBath'] + df['BsmtFullBath']
    
    df['RemodHome'] = (df['YearRemodAdd'] > df['YearBuilt']).apply(lambda x: 1 if x else 0)
    df['RemodGarage'] = (df['GarageYrBlt'] > df['YearBuilt']).apply(lambda x: 1 if x else 0)
    
    df['TotalPorchSF'] = df['OpenPorchSF'] + df['EnclosedPorch'] + df['3SsnPorch'] + df['ScreenPorch']
    df['TotalPorchDeckSF'] = df['OpenPorchSF'] + df['EnclosedPorch'] + df['3SsnPorch'] + df['ScreenPorch'] + df['WoodDeckSF']
    df['YardSF'] = df['LotArea'] - df['GrLivArea']
    
    df['Season'] = df['MoSold'].replace([12,1,2],"Winter").replace([3,4,5], "Spring").replace([6,7,8], "Summer").replace([9,10,11], "Fall")
    
    df['IndoorSF'] = df['1stFlrSF']+ df['2ndFlrSF'] + df['LowQualFinSF'] + df['TotalBsmtSF']
    df['AvgSFRm'] = (df['1stFlrSF'] + df['2ndFlrSF']) / df['TotRmsAbvGrd']
    
    df['MultiKitchen'] = df['KitchenAbvGr'].apply(lambda x: 1 if x > 1 else 0)
    df['MultiFireplace'] =  df['Fireplaces'].apply(lambda x: 1 if x > 1 else 0)
    df['ExtraRooms'] = df['TotRmsAbvGrd']- df['BedroomAbvGr'] - df['KitchenAbvGr']
    
    df['BsmtFinSF'] = df['BsmtFinSF1'] + df['BsmtFinSF2']
    df['BsmtFinYN'] = df['BsmtFinSF'].apply(lambda x: 1 if x > 0 else 0)
    
    df['PoolYN'] = df['PoolArea'].apply(lambda x: 1 if x > 0 else 0)
    df['FireplaceYN'] = df['Fireplaces'].apply(lambda x: 1 if x > 0 else 0)
    df['FenceYN'] = df['Fence'].notnull().apply(lambda x: 1 if x else 0)
    
    df['ExteriorMixedYN'] = (df['Exterior1st'] != df['Exterior2nd']).apply(lambda x: 1 if x else 0)  
    
    return df

#%% function fixes NAs by imputing 0, mean/mode or setting to "Unknown"/"None"
def nafix(df):    
    
    fixdf = df.copy()
    
    # specific oclumn drops
    fixdf = fixdf.drop(['PoolQC', 'MiscFeature', 'Alley', 'Fence', 'Id'], axis = 1)
      
    # specific quants
    fixdf['GarageYrBlt'].fillna(fixdf['YearRemodAdd'], inplace = True)
    
    # specific cats
    catnonecol = ['MasVnrType', 'FireplaceQu',
                  'BsmtQual','BsmtCond','BsmtExposure','BsmtFinType1','BsmtFinType2',
                  'GarageType','GarageFinish','GarageQual','GarageCond']
    
    for col in fixdf:
        if col in catnonecol:
            fixdf[col].fillna("None", inplace = True)
    
    # Catch the rest of NAs & split into cat/quant
    cdf = fixdf.drop(fixdf._get_numeric_data().columns, axis = 1)
    t = np.sum(cdf.isnull(), axis = 0) 
    cnacol = list(t[t>0].index.values)
    
    # If cat, then unknown. elseif quant, then 0.
    for col in fixdf:
        if col in cnacol:
            fixdf[col].fillna("Unknown", inplace = True)
        else:
            fixdf[col].fillna(0, inplace = True)
            
    return fixdf

#%% function converts select quant -> string (category)
def q_to_cat(ogdf, cols_to_convert):
    
    df = ogdf.copy()
    for col in df:
        if col in cols_to_convert:
            df[col] = df[col].astype(str)
    
    return df

#%% function does CATsuggested action
def optcat(ogdf, y, train_or_test):
    
    df = ogdf.copy()
    
    from CatAnalysis import catdf
    suggestdf = catdf(df, y, train_or_test)['suggest']
    
    quantifycols = list(suggestdf[suggestdf == "quantify"].index.values)
    ovacols = list(suggestdf[suggestdf == "1vA"].index.values)
    ignorecols = list(suggestdf[suggestdf == "ignore"].index.values)
    
    dumcols = list(suggestdf[suggestdf == 'dummify'].index.values)
    dumcols = dumcols + list(suggestdf[suggestdf == 'binary'].index.values)
    
    for col in df:
        if col in quantifycols:
            df[col] = df[col].fillna(0).replace('None', 0).replace('Po', 1).replace('Fa',2).replace('TA', 3).replace('Gd',4).replace('Ex',5)
        elif col in ovacols:
            df[col] = df[col].eq(df[col].mode()[0]).mul(1)
        elif col in ignorecols:
            df[col] = df.drop(col, axie = 1)
    
    df = pd.get_dummies(df, columns = dumcols, drop_first = True)
    
    return df

#%% Feature Manipulation GO!!!!
    
def thewholeshabang(ogdf, y, tot):
    temp = beefup(ogdf)
    temp = q_to_cat(temp, ['MSSubClass', 'OverallCond', 'MoSold'])
    temp = nafix(temp)
    temp = optcat(temp, y, tot)
    return temp


