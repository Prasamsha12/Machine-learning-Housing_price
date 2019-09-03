# -*- coding: utf-8 -*-
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler



test = pd.read_csv('./test.csv')
data = pd.read_csv('./train.csv')
print(data.shape)
print(test.shape)
#print(data.isnull().values.any())
#print(data.isnull().sum())

#print(test.columns)
#pd.options.display.max_rows = 500
#converting nan values with median for train data
median = data['LotFrontage'].median()
data['LotFrontage'].fillna(median, inplace=True)
#median1 = data['GarageYrBlt'].median()
#data['GarageYrBlt'].fillna(median1, inplace=True)
median3 = data['MasVnrArea'].median()
data['MasVnrArea'].fillna(median3, inplace=True)

#converting nan values with median for test data
median = test['LotFrontage'].median()
test['LotFrontage'].fillna(median, inplace=True)
#median1 = test['GarageYrBlt'].median()
#test['GarageYrBlt'].fillna(median1, inplace=True)
median3 = test['MasVnrArea'].median()
test['MasVnrArea'].fillna(median3, inplace=True)

#transforming nan values of string and string to numerical for train data
categorical_values = ["SaleCondition","SaleType","MiscFeature","Fence","PoolQC",'Street','CentralAir',"PavedDrive","GarageCond","GarageQual","GarageFinish","GarageType","FireplaceQu","Functional","KitchenQual","Electrical","HeatingQC","Heating","BsmtFinType2","BsmtFinType1","BsmtExposure","BsmtCond","BsmtQual","Foundation","ExterCond","ExterQual","MasVnrType","Exterior2nd","Exterior1st","RoofMatl","RoofStyle","HouseStyle","BldgType","Condition2","Condition1","Neighborhood","LandSlope","LotConfig","Utilities","LandContour","LotShape","Alley","MSZoning","MSSubClass"]
data = pd.get_dummies(data,columns=categorical_values)
#print(data.isnull().sum())

#transforming nan values of string and string to numerical for test data
categorical_values = ["BsmtFullBath","BsmtHalfBath","SaleCondition","SaleType","MiscFeature","Fence","PoolQC",'Street','CentralAir',"PavedDrive","GarageCond","GarageQual","GarageFinish","GarageType","FireplaceQu","Functional","KitchenQual","Electrical","HeatingQC","Heating","BsmtFinType2","BsmtFinType1","BsmtExposure","BsmtCond","BsmtQual","Foundation","ExterCond","ExterQual","MasVnrType","Exterior2nd","Exterior1st","RoofMatl","RoofStyle","HouseStyle","BldgType","Condition2","Condition1","Neighborhood","LandSlope","LotConfig","Utilities","LandContour","LotShape","Alley","MSZoning","MSSubClass"]
test = pd.get_dummies(test,columns=categorical_values)
print(test.isnull().sum())

#replacing NAN with 0 for traindata

data['GarageYrBlt'].fillna(0, inplace=True)

#replacing NAN with 0 for test data
test['GarageYrBlt'].fillna(0, inplace=True)
test['GarageCars'].fillna(0, inplace=True)
test['GarageArea'].fillna(0, inplace=True)
test['TotalBsmtSF'].fillna(0, inplace=True)
test['BsmtFinSF1'].fillna(0, inplace=True)
test['BsmtFinSF2'].fillna(0, inplace=True)
test['BsmtUnfSF'].fillna(0, inplace=True)
#print(test.isnull().sum())


X = data.drop("SalePrice", axis=1)
y = data.SalePrice




##Allign test with original data

extra_columns = list(set(X.columns.values) - set(test.columns.values))
X = X.drop(extra_columns, axis=1)
extra_columns = list(set(test.columns.values) - set(X.columns.values))
test = test.drop(extra_columns, axis=1)

#feature scaling
sc = StandardScaler()
X_train = sc.fit_transform(X)
X_test = sc.fit_transform(test)
#training the algorithm
regressor = RandomForestRegressor(n_estimators=100,random_state=0)
regressor = regressor.fit(X_train,y)



#Making predictions on the test data
y_pred = regressor.predict(X_test)
y_pred_df = pd.DataFrame(y_pred, columns =["SalePrice"])
y_pred_df.index = y_pred_df.index + 1461
y_pred_df.to_csv("y_pred.csv")
print(y_pred_df.shape)



