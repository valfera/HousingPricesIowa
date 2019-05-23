import pandas as pd
import numpy as np
from scipy import stats
from matplotlib import pyplot as plt
from sklearn.linear_model import LinearRegression

train = pd.read_csv("train.csv")
train1 = pd.read_csv('train.csv')

test = pd.read_csv("test.csv")
test1 = pd.read_csv("test.csv")

# Deal With Missing Values

missingCols = train.columns[train.isnull().any()]
print('The columns with missingness are %s' %(missingCols))

missingCols = test.columns[test.isnull().any()]
print('The columns with missingness are %s' %(missingCols))

#LotFrontage

# train.LotFrontage.unique()

train['LotFrontage'].fillna(train['LotFrontage'].mean(), inplace = True)

test['LotFrontage'].fillna(test['LotFrontage'].mean(), inplace = True)

#Alley

# train.Alley.unique()

train['Alley'] = train['Alley'].replace(np.nan, 'None', regex=True)

#MasVnrType

# train.MasVnrType.unique()

train['MasVnrType'] = train['MasVnrType'].replace(np.nan, 'None', regex=True)

#MasVnrArea

# train.MasVnrArea.unique()

train['MasVnrArea'].fillna(train['MasVnrArea'].mean(), inplace = True)

test['MasVnrArea'].fillna(test['MasVnrArea'].mean(), inplace = True)

#BsmtQual

# train.BsmtQual.unique()

train['BsmtQual'] = train['BsmtQual'].replace(np.nan, 'None', regex=True)

#BsmtCond

# train.BsmtCond.unique()

train['BsmtCond'] = train['BsmtCond'].replace(np.nan, 'None', regex=True)

#BsmtExposure

# train.BsmtExposure.unique()

train['BsmtExposure'] = train['BsmtExposure'].replace(np.nan, 'None', regex=True)

#BsmtFinSF1

# test.BsmtFinSF1.unique()

test['BsmtFinSF1'].fillna(test['BsmtFinSF1'].mean(), inplace = True)

#BsmtFinSF1

# test.BsmtFinSF2.unique()

test['BsmtFinSF2'].fillna(test['BsmtFinSF2'].mean(), inplace = True)

#BsmtUnfSF

# train.BsmtUnfSF.unique()

test['BsmtUnfSF'].fillna(test['BsmtUnfSF'].mean(), inplace = True)

#TotalBsmtSF

# train.TotalBsmtSF.unique()

test['TotalBsmtSF'].fillna(test['TotalBsmtSF'].mean(), inplace = True)

#BsmtFullBath

# test.BsmtFullBath.unique()

test['BsmtFullBath'].fillna(test['BsmtFullBath'].mean(), inplace = True)

#BsmtHalfBath

# test.BsmtHalfBath.unique()

test['BsmtHalfBath'].fillna(test['BsmtHalfBath'].mean(), inplace = True)

#BsmtFinType1

# train.BsmtFinType1.unique()

train['BsmtFinType1'] = train['BsmtFinType1'].replace(np.nan, 'None', regex=True)

#BsmtFinType2

# train.BsmtFinType2.unique()

train['BsmtFinType2'] = train['BsmtFinType2'].replace(np.nan, 'None', regex=True)

#Electrical

# train.Electrical.unique()

train['Electrical'] = train['Electrical'].replace(np.nan, 'None', regex=True)

#FireplaceQu

# train.FireplaceQu.unique()

train['FireplaceQu'] = train['FireplaceQu'].replace(np.nan, 'None', regex=True)

#GarageType

# train.GarageType.unique()

train['GarageType'] = train['GarageType'].replace(np.nan, 'None', regex=True)

#GarageYrBlt

# train.GarageYrBlt.unique()

train['GarageYrBlt'].fillna(train['GarageYrBlt'].mode(), inplace = True)

test['GarageYrBlt'].fillna(test['GarageYrBlt'].mode(), inplace = True)

#GarageFinish

# train.GarageFinish.unique()

train['GarageFinish'] = train['GarageFinish'].replace(np.nan, 'None', regex=True)

#GarageQual

# train.GarageQual.unique()

train['GarageQual'] = train['GarageQual'].replace(np.nan, 'None', regex=True)

#GarageCond

# train.GarageCond.unique()

train['GarageCond'] = train['GarageCond'].replace(np.nan, 'None', regex=True)

#GarageCars

# test.GarageCars.unique()

test['GarageCars'].fillna(test['GarageCars'].mean(), inplace = True)

#GarageArea

# test.GarageArea.unique()

test['GarageArea'].fillna(test['GarageArea'].mean(), inplace = True)

#PoolQC

# train.PoolQC.unique()

train['PoolQC'] = train['PoolQC'].replace(np.nan, 'None', regex=True)

#Fence

# train.Fence.unique()

train['Fence'] = train['Fence'].replace(np.nan, 'None', regex=True)

#MiscFeature

# train.MiscFeature.unique()

train['MiscFeature'] = train['MiscFeature'].replace(np.nan, 'None', regex=True)

# Feature Engineering

train['TotalSF'] = train['TotalBsmtSF'] + train['1stFlrSF'] + train['2ndFlrSF']

test['TotalSF'] = test['TotalBsmtSF'] + test['1stFlrSF'] + test['2ndFlrSF']

#Note train['BsmtUnfSF'] + train['BsmtFinSF1'] + train['BsmtFinSF2']

train['TotalBath'] = (train['FullBath'] + 0.5*train['HalfBath'])

test['TotalBath'] = (test['FullBath'] + 0.5*test['HalfBath'])

train['TotalPorch'] = train['WoodDeckSF'] + train['OpenPorchSF'] + train['EnclosedPorch'] + train['3SsnPorch'] + train['ScreenPorch']

test['TotalPorch'] = test['WoodDeckSF'] + test['OpenPorchSF'] + test['EnclosedPorch'] + test['3SsnPorch'] + test['ScreenPorch']

train = train.drop(['BsmtUnfSF', 'BsmtFinSF1', 'BsmtFinSF2', 'TotalBsmtSF', '1stFlrSF', '2ndFlrSF', 'FullBath', 'HalfBath', 'WoodDeckSF', 'OpenPorchSF', 'EnclosedPorch', '3SsnPorch', 'ScreenPorch'], axis = 1)

test = test.drop(['BsmtUnfSF', 'BsmtFinSF1', 'BsmtFinSF2', 'TotalBsmtSF', '1stFlrSF', '2ndFlrSF', 'FullBath', 'HalfBath', 'WoodDeckSF', 'OpenPorchSF', 'EnclosedPorch', '3SsnPorch', 'ScreenPorch'], axis = 1)

# Get Dummies

train = pd.get_dummies(train, columns = ['MSSubClass', 'MSZoning', 'Street', 'Alley', 'LotShape', 'LandContour', 'Utilities', 'LotConfig', 'LandSlope', 'Neighborhood', 'Condition1', 'Condition2', 'BldgType', 'HouseStyle', 'RoofStyle', 'RoofMatl', 'Exterior1st', 'Exterior2nd', 'MasVnrType', 'ExterQual', 'ExterCond', 'Foundation', 'BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2', 'Heating', 'HeatingQC', 'CentralAir', 'Electrical', 'KitchenQual', 'Functional', 'FireplaceQu', 'GarageType', 'GarageFinish', 'GarageQual', 'GarageCond', 'PavedDrive', 'PoolQC','Fence', 'MiscFeature', 'SaleType', 'SaleCondition'], drop_first = True)

test = pd.get_dummies(test, columns = ['MSSubClass', 'MSZoning', 'Street', 'Alley', 'LotShape', 'LandContour', 'Utilities', 'LotConfig', 'LandSlope', 'Neighborhood', 'Condition1', 'Condition2', 'BldgType', 'HouseStyle', 'RoofStyle', 'RoofMatl', 'Exterior1st', 'Exterior2nd', 'MasVnrType', 'ExterQual', 'ExterCond', 'Foundation', 'BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2', 'Heating', 'HeatingQC', 'CentralAir', 'Electrical', 'KitchenQual', 'Functional', 'FireplaceQu', 'GarageType', 'GarageFinish', 'GarageQual', 'GarageCond', 'PavedDrive', 'PoolQC','Fence', 'MiscFeature', 'SaleType', 'SaleCondition'], drop_first = True)

# Feature Reduction

Remove Features Lowly Correlated With Sale Price

cormat = train.corr()

hcorr = cormat['SalePrice'].loc[cormat['SalePrice'] > .3]

hcorr = pd.DataFrame(hcorr)

hcorr = hcorr.index.tolist()

train = train.reindex(hcorr, axis = 1)

test = test.reindex(hcorr, axis = 1)

train = train.drop('SalePrice', axis = 1)

test = test.drop('SalePrice', axis = 1)

len(hcorr)

train.shape

test.shape

Drop Highly Correlated Features

corrs = train.corr().values

def remove_high_corrs(corrs, thresh):
    idx = []
    for i in range(len(corrs)):
        if not np.any(corrs[i, :i] > thresh):
            idx.append(i)  
    return idx

var_to_keep = remove_high_corrs(corrs, 0.7)

len(train.columns)

len(var_to_keep)

train = train.iloc[:, var_to_keep]

test = test.iloc[:, var_to_keep]

train.shape

test.shape

# Distribution of Features

plots = train.hist(bins = 50, figsize = (15, 15))

# Prepare Model

y = train1.SalePrice

lm = LinearRegression()
lm.fit(train, y)

p = lm.predict(test)

ind = pd.DataFrame(test1.Id)

predictions = pd.DataFrame(p, columns = ['SalePrice'], index = None)

submission = pd.concat([ind, predictions], axis = 1)

submission.head()

pd.DataFrame(submission).to_csv("predictions.csv", index = None)