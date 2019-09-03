# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score
import pickle
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2

data = pd.read_csv('train_set.csv', index_col='Id')
print(data.isnull().values.any())
print(data.isnull().sum())
print(data.shape)

#converting nan values with median
median = data['LotFrontage'].median()
data['LotFrontage'].fillna(median, inplace=True)
#median1 = data['GarageYrBlt'].median()
#data['GarageYrBlt'].fillna(median1, inplace=True)
median3 = data['MasVnrArea'].median()
data['MasVnrArea'].fillna(median3, inplace=True)

#replacing NAN with 0

data['GarageYrBlt'].fillna(0, inplace=True)

#transforming nan values of string and string to numerical
categorical_values = ["SaleCondition","SaleType","MiscFeature","Fence","PoolQC","PavedDrive","GarageCond","GarageQual","GarageFinish","GarageType","FireplaceQu","Functional","KitchenQual","Electrical","HeatingQC","Heating","BsmtFinType2","BsmtFinType1","BsmtExposure","BsmtCond","BsmtQual","Foundation","ExterCond","ExterQual","MasVnrType","Exterior2nd","Exterior1st","RoofMatl","RoofStyle","HouseStyle","BldgType","Condition2","Condition1","Neighborhood","LandSlope","LotConfig","Utilities","LandContour","LotShape","Alley","MSZoning","MSSubClass"]
data = pd.get_dummies(data,columns=categorical_values)
#using encoder for binary values
encoding= ['Street','CentralAir']
enc = LabelEncoder()
for e in encoding:
    data[e]= enc.fit_transform(data[e])
 
#dropping the saleprice column and assigning it in y

X = data.drop("SalePrice", axis=1)
y = data.SalePrice

#featureselection
#apply SelectKBest class to extract top 10 best features
bestfeatures = SelectKBest(score_func=chi2, k=10)
fit = bestfeatures.fit(X,y)
dfscores = pd.DataFrame(fit.scores_)
dfcolumns = pd.DataFrame(X.columns)
#concat two dataframes for better visualization 
featureScores = pd.concat([dfcolumns,dfscores],axis=1)
featureScores.columns = ['Specs','Score']  #naming the dataframe columns
print(featureScores.nlargest(10,'Score'))  #print 10 best features
#print(data.columns)



#dividing data into train and test
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size = 0.2,random_state=0)

#feature scaling
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)
#training the algorithm
regressor = RandomForestRegressor(n_estimators=80,random_state=0)
regressor = regressor.fit(X_train,y_train)
#Making predictions on the test data
y_pred = regressor.predict(X_test)

#comparing prediction and actual values
data_1 = pd.DataFrame({'Actual':y_test, 'Predicted':y_pred})

#evaluating the algorithm
print('Mean absolute error:',metrics.mean_absolute_error(y_test,y_pred))
print('Mean squared error:',metrics.mean_squared_error(y_test,y_pred))
print('Root Mean squared error:',np.sqrt(metrics.mean_squared_error(y_test,y_pred)))
#r2 score
score = r2_score(y_test, y_pred)
print(score)

# save the model to disk
filename = 'finalized_model.csv'
pickle.dump(regressor, open(filename, 'wb'))

# load the model from disk
loaded_model = pickle.load(open(filename, 'rb'))
result = loaded_model.score(X_test, y_test)
print(result)





