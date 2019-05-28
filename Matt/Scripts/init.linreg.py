import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn import linear_model


#%% Import Data
train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')

#%lowercase for my brain
train.columns = [x.lower() for x in train.columns]
test.columns = [x.lower() for x in test.columns]


#outlier removal
train = train[train.garagearea < 1200]
train = train[train.totalbsmtsf < 2500]

#street type to street bool
train['enc_street'] = pd.get_dummies(train.street, drop_first = True)
test['enc_street'] = pd.get_dummies(test.street, drop_first = True)

#pool type to pool bool
def encode(x): return 1 if x > 0 else 0
train['enc_pool'] = train.poolarea.apply(encode)
test['enc_pool'] = train.poolarea.apply(encode)

#financial crisis bool
def encode(x): return 1 if x > 2008 else 0
train['enc_after08'] = train.yrsold.apply(encode)
test['enc_after08'] = train.yrsold.apply(encode)

#condition-partial bool
def encode(x): return 1 if x == 'Partial' else 0
train['enc_condition'] = train.salecondition.apply(encode)
test['enc_condition'] = test.salecondition.apply(encode)

#interpolate
data = train.select_dtypes(include=[np.number]).interpolate().dropna()


##################
y = np.log(train.saleprice)
X = data.drop(['saleprice', 'id','poolarea', 'yrsold'], axis = 1) 


X_train, X_test, y_train, y_test = train_test_split(X, y, random_state = 100, test_size = .2)

lm = linear_model.LinearRegression()

lm.fit(X_train,  y_train)

lm.score(X_test, y_test)


df = pd.DataFrame()
df['Id'] = test.id

features = test.select_dtypes(include = [np.number]).drop(['id','poolarea', 'yrsold'], axis = 1).interpolate()

predictions = lm.predict(features)
df['SalePrice'] = final_lm_predictions

df.to_csv('init.simple_linreg.csv')