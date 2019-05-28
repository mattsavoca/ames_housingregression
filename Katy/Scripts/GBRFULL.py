# -*- coding: utf-8 -*-
"""
GBR - first attempt
Using only quant values
"""
# Basic imports
import numpy as np
import pandas as pd
train = pd.read_csv('./train.csv')
test_r = pd.read_csv('./test.csv')
df = train.copy()

#%% Function to run gbr
def gbrrun(df, logyn):
    
    import time
    stime = time.time()

    # set x & y
    x = df.drop('SalePrice', axis = 1).fillna(0) 
    y = np.log(df['SalePrice']) if logyn else df['SalePrice']    
    
    # Test/Train split
    from sklearn.model_selection import train_test_split
    xtrain, xtest, ytrain, ytest = train_test_split(x, y, test_size = 0.2, random_state = 88)
    
    # Create GBR object
    from sklearn import ensemble
    #params = {'n_estimators': 500, 'max_depth': 4, 'min_samples_split': 2, 'learning_rate': 0.01, 'loss': 'ls'}
    #clf = ensemble.GradientBoostingRegressor(**params)
    #clf.fit(xtrain, ytrain)
    
    # Show results - "baseline"
    #score = round(clf.score(xtest, ytest),4)
    #meanerr =  round(np.mean(abs(clf.predict(xtest) - ytest)),6)
    #accuracy = round((1 - np.mean(abs(clf.predict(xtest) - ytest)/ytest))*100,4)
    #print("Score: " + str(score) + "\nMean Err: $" + str(meanerr) + "\nAccuracy: " + str(accuracy))
    
    # CV for better parameters
    # https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.GradientBoostingRegressor.html
    from sklearn.model_selection import GridSearchCV
    gbr = ensemble.GradientBoostingRegressor()
    grid = {#'n_estimators':[100, 500], # THIS TAKES FOREVER... when we do the real thing,  just do n = 500+. more trees is better.
            'learning_rate':[0.1],
            'max_depth':[4,7,10],
            'max_features':[6,9,12,15], # Choosing max_features < n_features leads to a reduction of variance and an increase in bias
            'subsample':[.5,.75,1], # choosing subsample < 1 leads to a reduction of variance and an increase in bias
            'random_state':[1]}
    grid_search = GridSearchCV( estimator=gbr, param_grid=grid, scoring='neg_mean_squared_error', n_jobs=1, cv=5, verbose=5) #Takes 30s
    grid_search.fit(x, y)
    bestparams = grid_search.best_params_ 
    
    # Run again with BEST parameters
    bestmodel = grid_search.best_estimator_
    bestscore = round(bestmodel.score(xtest, ytest),4).mean()
    besterr = round(np.mean(abs(bestmodel.predict(xtest) - ytest)),4)
    bestacc = round((1 - np.mean(abs(bestmodel.predict(xtest) - ytest)/ytest))*100,4)
   #print("Score: " + str(bestscore) + "\nMean Err: $" + str(besterr) + "\nAccuracy: " + str(bestacc))
    
    return bestscore, besterr, bestacc, time.time() - stime, bestparams


#%% Trying all models
    
# Create 3 datasets for model comparison
from FeatureAdd import nafix, thewholeshabang, beefup, q_to_cat
df_nocat = train.copy() ; df_nocat = df_nocat._get_numeric_data().fillna(0) 
df_qcat = train.copy() ; df_qcat = nafix(df_qcat)
df_dumcat = train.copy() ; df_dumcat = nafix(df_dumcat); df_dumcat = pd.get_dummies(df_dumcat, drop_first = True)
df_optcat = train.copy() ; df_optcat = thewholeshabang(df_optcat, df_optcat['SalePrice'], "train")
df_mattcat = pd.read_csv('./train_preproc.csv').drop(['Unnamed: 0', 'saleprice', 'log_price'], axis = 1)
df_mattcat = pd.concat([df_mattcat, df['SalePrice']], axis=1)
df_mattcat = pd.get_dummies(df_mattcat, drop_first = True)

# Setup ResultsDf to store various model results
resultsdf = pd.DataFrame(columns = ['Score', 'MeanErr', 'Accuracy', 'CompTime', 'BestParams'])

# Run with NOCAT (38f) - dropping all categories
xscore, xerr, xacc, xtime, xparams = gbrrun(df_nocat, logyn = False)
resultsdf.loc["NOcat"] = [xscore, xerr, xacc, xtime, xparams]
xscore, xerr, xacc, xtime, xparams = gbrrun(df_nocat, logyn = True)
resultsdf.loc["NOcat_log"] = [xscore, xerr, xacc, xtime, xparams]

# Run with QCAT (46f) - C->Q for Poor-Excellent categories, and then dropping all categories
quantcol = ['BsmtQual', 'BsmtCond', 'KitchenQual', 'ExterQual', 'ExterCond', 'GarageQual', 'GarageCond', 'HeatingQC', 'FireplaceQu', 'PoolQC']
for col in df_qcat:
    if col in quantcol:
        df_qcat[col] = df_qcat[col].replace('None', 0).replace('Po', 1).replace('Fa',2).replace('TA', 3).replace('Gd',4).replace('Ex',5)
df_qcat = df_qcat._get_numeric_data()

xscore, xerr, xacc, xtime, xparams = gbrrun(df_qcat, logyn = False)
resultsdf.loc["Qcat"] = [xscore, xerr, xacc, xtime, xparams]
xscore, xerr, xacc, xtime, xparams = gbrrun(df_qcat, logyn = True)
resultsdf.loc["Qcat_log"] = [xscore, xerr, xacc, xtime, xparams]

# Run with DUMCAT (249f) - dummifying EVERYTHING
xscore, xerr, xacc, xtime, xparams = gbrrun(df_dumcat, logyn = False)
resultsdf.loc["DUMcat"] = [xscore, xerr, xacc, xtime, xparams]
xscore, xerr, xacc, xtime, xparams = gbrrun(df_dumcat, logyn = True)
resultsdf.loc["DUMcat_log"] = [xscore, xerr, xacc, xtime, xparams]

# Run with OPTCAT (242f)- After hours of new feature adds, algorithms to decide what to do with each column, etc...
xscore, xerr, xacc, xtime, xparams = gbrrun(df_optcat, logyn = False)
resultsdf.loc["KATcat"] = [xscore, xerr, xacc, xtime, xparams]
xscore, xerr, xacc, xtime, xparams = gbrrun(df_optcat, logyn = True)
resultsdf.loc["KATcat_log"] = [xscore, xerr, xacc, xtime, xparams]

# Run with MATTCAT
xscore, xerr, xacc, xtime, xparams = gbrrun(df_mattcat, logyn = False)
resultsdf.loc["MATTcat"] = [xscore, xerr, xacc, xtime, xparams]
xscore, xerr, xacc, xtime, xparams = gbrrun(df_mattcat, logyn = True)
resultsdf.loc["MATTcat_log"] = [xscore, xerr, xacc, xtime, xparams]

# Save Resultsdf
resultsdf.to_csv('GBR_ModelCompare2.csv', index=True)

#%% Run model on test data

# Get test data?
test = pd.read_csv('./test_preproc.csv')
dftest = test.copy().drop('Unnamed: 0', axis = 1)
dftest = pd.get_dummies(dftest, drop_first = True)

# Make tree with best params & set n_estimators to 1000
from sklearn import ensemble
from sklearn.model_selection import GridSearchCV
gbr = ensemble.GradientBoostingRegressor(**xparams)
gbr.set_params(n_estimators = 1000)
gbr.get_params
x = df_mattcat.drop('SalePrice', axis = 1)
y = np.log(df_mattcat['SalePrice'])
gbr.fit(x,y)

# Make sure columns match
'''
diffcol = list(set(list(dftest.columns)) - set(list(x.columns)))
len(diffcol)
diffcol
dftest = dftest.drop(diffcol, axis = 1)

diffcol2 = list(set(list(x.columns)) - set(list(dftest.columns)))
len(diffcol2)
diffcol2
x = x.drop(diffcol2, axis = 1)
'''

submission = pd.Series(gbr.predict(dftest))
submission = pd.concat([test_r['Id'], submission], axis = 1)
submission.columns = ['Id','SalePrice']
submission['SalePrice'] = round(np.exp(submission['SalePrice']),2) 

submission.to_csv("GBR_MattCAT_Submission.csv", index = False) # change name accordingly!



