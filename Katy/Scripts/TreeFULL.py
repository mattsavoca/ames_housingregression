# -*- coding: utf-8 -*-
"""
Tree FULL
"""

# Basic imports
import numpy as np
import pandas as pd
train = pd.read_csv('./train.csv')

# Create 3 datasets for model comparison
from FeatureAdd import nafix, thewholeshabang, beefup, q_to_cat
df_dumcat = train.copy() ; df_dumcat = nafix(df_dumcat); df_dumcat = pd.get_dummies(df_dumcat, drop_first = True)
df_nocat = train.copy() ; df_nocat = df_nocat._get_numeric_data().fillna(0) 
df_optcat = train.copy() ; df_optcat = thewholeshabang(df_optcat, df_optcat['SalePrice'], "train")
df_qcat = train.copy() ; df_qcat = nafix(df_qcat)

# Setup ResultsDf to store various model results
resultsdf = pd.DataFrame(columns = ['Score', 'Mean Abs Err', 'Comptime', 'MaxFeatures', 'Top5'])

# Run with NOCAT (38f) - dropping all categories
xscore, xerr, xtime, xmaxf, xtop5 = treerun(df_nocat, logyn = False)
resultsdf.loc["Nocat"] = [xscore, xerr, xtime, xmaxf, xtop5]
xscore, xerr, xtime, xmaxf, xtop5 = treerun(df_nocat, logyn = True)
resultsdf.loc["Nocat_log"] = [xscore, xerr, xtime, xmaxf, xtop5]

# Run with QCAT (46f) - C->Q for Poor-Excellent categories, and then dropping all categories
quantcol = ['BsmtQual', 'BsmtCond', 'KitchenQual', 'ExterQual', 'ExterCond', 'GarageQual', 'GarageCond', 'HeatingQC', 'FireplaceQu', 'PoolQC']
for col in df_qcat:
    if col in quantcol:
        df_qcat[col] = df_qcat[col].replace('None', 0).replace('Po', 1).replace('Fa',2).replace('TA', 3).replace('Gd',4).replace('Ex',5)
df_qcat = df_qcat._get_numeric_data()

xscore, xerr, xtime, xmaxf, xtop5 = treerun(df_qcat, logyn = False)
resultsdf.loc["Qcat"] = [xscore, xerr, xtime, xmaxf, xtop5]
xscore, xerr, xtime, xmaxf, xtop5 = treerun(df_qcat, logyn = True)
resultsdf.loc["Qcat_log"] = [xscore, xerr, xtime, xmaxf, xtop5]

# Run with DUMCAT (249f) - dummifying EVERYTHING
xscore, xerr, xtime, xmaxf, xtop5 = treerun(df_dumcat, logyn = False)
resultsdf.loc["Dumcat"] = [xscore, xerr, xtime, xmaxf, xtop5]
xscore, xerr, xtime, xmaxf, xtop5 = treerun(df_dumcat, logyn = True)
resultsdf.loc["Dumcat_log"] = [xscore, xerr, xtime, xmaxf, xtop5]

# Run with OPTCAT (242f)- After hours of new feature adds, algorithms to decide what to do with each column, etc...
xscore, xerr, xtime, xmaxf, xtop5 = treerun(df_optcat, logyn = False)
resultsdf.loc["Optcat"] = [xscore, xerr, xtime, xmaxf, xtop5]
xscore, xerr, xtime, xmaxf, xtop5 = treerun(df_optcat, logyn = True)
resultsdf.loc["Optcat_log"] = [xscore, xerr, xtime, xmaxf, xtop5]

# Save Resultsdf
resultsdf.to_csv('RandomForest_ModelCompare.csv', index=True)

#%% Function to run Random Forest
def treerun(df, logyn):
    
    import time
    import math
    stime = time.time()
    
    x = df.drop('SalePrice', axis = 1)
    y = np.log(df['SalePrice']) if logyn else df['SalePrice']
    cols = list(x.columns)
    
    # Tree Setup
    from sklearn import ensemble
    rf = ensemble.RandomForestRegressor()
    rf.set_params(random_state = 0, n_estimators = 100, max_features = round(math.sqrt(df.shape[1]),0)) #sqrt(numcol)
    
    # Test/Train split
    from sklearn.model_selection import train_test_split
    xtrain, xtest, ytrain, ytest = train_test_split(x, y, test_size = 0.2, random_state = 88)
         
    # CV to optimize max features & num trees
    from sklearn.model_selection import GridSearchCV
    grid = [{ "max_features": range(1, 25)}]
    grid_search = GridSearchCV(rf, grid, cv=5, n_jobs=-1)
    grid_search.fit(x, y)
 
    # Run again with BEST parameters
    bestrf = grid_search.best_estimator_
    bestscore = round(bestrf.score(xtest, ytest),4).mean()
    besterr = round(np.mean(abs(bestrf.predict(xtest) - ytest)),2)   
    
    # Rank importance
    rank = list(bestrf.feature_importances_)
    rank = [(x, round(rank, 2)) for x, rank in zip(cols, rank)]
    rank = sorted(rank, key = lambda x: x[1], reverse = True)
        
    # return score, mean abs error, top 5, time
    return bestscore, besterr, time.time() - stime, grid_search.best_params_['max_features'], rank[:5]

#%% Code to use model on test!
    
test = pd.read_csv("test.csv")
df = test.copy(); df.set_index('Id')

# Go through same feature engineering as train set
df = beefup(df)
df = nafix(df) 
df = q_to_cat(df, ['MSSubClass', 'OverallCond', 'MoSold'])

# Take the suggest list from previosly ran CatAnalysis for TRAIN set
suggestdf = pd.read_csv("FeatureSuggestion.csv")
suggestdf = suggestdf.set_index('Unnamed: 0')['suggest']

# split into different feature sets     
quantifycols = list(suggestdf[suggestdf == "quantify"].index.values)
ovacols = list(suggestdf[suggestdf == "1vA"].index.values)
ignorecols = list(suggestdf[suggestdf == "ignore"].index.values)
dumcols = list(suggestdf[suggestdf == 'dummify'].index.values)
dumcols = dumcols + list(suggestdf[suggestdf == 'binary'].index.values)

# perform the suggested action
for col in df:
    if col in quantifycols:
        df[col] = df[col].fillna(0).replace('None', 0).replace('Po', 1).replace('Fa',2).replace('TA', 3).replace('Gd',4).replace('Ex',5)
    elif col in ovacols:
        df[col] = df[col].eq(df[col].mode()[0]).mul(1)
    elif col in ignorecols:
        df[col] = df.drop(col, axis = 1)

df = pd.get_dummies(df, columns = dumcols, drop_first = True)

# Now run transformed DF through tree
ytrain = np.log(df_optcat['SalePrice'])
xtrain = df_optcat.drop('SalePrice', axis = 1)
xtest = test

# with just dummy all
ytrain = df_dumcat['SalePrice']
xtrain = df_dumcat.drop(['SalePrice', 'Id'], axis = 1)
df = nafix(df)
xtest = pd.get_dummies(df, drop_first = True)


# Tree Setup
from sklearn import ensemble
rf = ensemble.RandomForestRegressor()
rf.set_params(random_state = 0, n_estimators = 100, max_features = 23) # as per results 
rf.fit(xtrain, ytrain)

# Predict!
submission = pd.Series(rf.predict(xtest))
submission = pd.concat([test['Id'], submission], axis = 1)
submission.columns = ['Id','SalePrice']
submission['SalePrice'] = round(np.exp(submission['SalePrice']),2)

submission.to_csv("RFDUM_Submission.csv", index = False)

# which columns are missing?
diffcol = list(set(list(xtrain.columns)) - set(list(xtest.columns)))
len(list(set(list(xtrain.columns)) - set(list(xtest.columns))))
for i in diffcol[:10]:
    xtest[i] = 0

