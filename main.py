import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import warnings
from scipy import stats
from scipy.special import boxcox1p
from sklearn.model_selection import KFold, cross_val_score
from sklearn.preprocessing import LabelEncoder
warnings.filterwarnings('ignore')

# Import Dataset
train = pd.read_csv("../input/train.csv")
test = pd.read_csv("../input/test.csv")

print("train shape:", train.shape)
print("test shape:", test.shape)
%matplotlib inline


# First Look
train.head()
train.info()
train.describe()
train.isnull().sum()
train['OverallQual'].unique()
train['OverallQual'].value_counts()
train.groupby('OverallQual').count()
train[train['OverallQual']>9]


# Visual Analysis

# Correlation map
corrmat = train.corr()
plt.subplots(figsize=(16, 9))
sns.heatmap(corrmat, vmax = 0.8, square=True)

# Target Value Analysis
train['SalePrice'].describe()

sns.distplot(train['SalePrice'])

Skewness = f"Skewness: {train['SalePrice'].skew():.4f}"
Kurtosis = f"Kurtosis: {train['SalePrice'].kurt():.4f}"
print(Skewness, Kurtosis)


# Data Processing

numberical_features=train.select_dtypes(include=[np.number])
categorical_features=train.select_dtypes(include=[np.object])

# save ID and drop it from train
train_ID=train['Id']
test_ID=test['Id']
train.drop('Id', axis=1, inplace=True)
test.drop('Id', axis=1, inplace=True)

# Noice Filtering
fig, ax = plt.subplots()
ax.scatter(x = train['GrLivArea'], y = train['SalePrice'])
plt.xlabel('GrLiveArea')
plt.ylabel('SalePrice')
plt.show()

train = train.drop(train[(train['GrLivArea']>4000) & (train['SalePrice']<300000)].index)

# Replot After Filtering
fig, ax = plt.subplots()
ax.scatter(x = train['GrLivArea'], y = train['SalePrice'])
plt.xlabel('GrLiveArea')
plt.ylabel('SalePrice')
plt.show()

sns.distplot(train['SalePrice'] , fit= stats.norm);
(mu, sigma) = stats.norm.fit(train['SalePrice'])

# Plot the Distribution
plt.legend(['Normal dist. ($\mu=$ {:.2f} and $\sigma=$ {:.2f} )'.format(mu, sigma)],loc='best')
plt.ylabel('Frequency')
plt.title('SalePrice distribution')
plt.show()

# apply log(1+x) to all elements of the column
train['SalePrice'] = np.log1p(train['SalePrice'])

# new distribution 
sns.distplot(train['SalePrice'] , fit= stats.norm);
(mu, sigma) = stats.norm.fit(train['SalePrice'])

# Now plot the distribution
plt.legend(['Normal dist. ($\mu=$ {:.2f} and $\sigma=$ {:.2f} )'.format(mu, sigma)],loc='best')
plt.ylabel('Frequency')
plt.title('SalePrice distribution')
plt.show()


# Input Missing data

# Concatenate the train and test data
n_train = train.shape[0]
n_test = test.shape[0]
y_train = train['SalePrice'].values
all_data = pd.concat((train, test)).reset_index(drop = True)
all_data.drop(['SalePrice'], axis = 1, inplace = True)
all_data.shape

# Find all missing data
all_data_na = (all_data.isnull().sum()/len(all_data)) * 100
all_data_na = all_data_na.drop(all_data_na[all_data_na == 0].index).sort_values(ascending = False)
missing_data = pd.DataFrame({'missing ratio' : all_data_na})
print(len(missing_data))
print(missing_data)

# Fill in missing data
all_data['PoolQC'] = all_data['PoolQC'].fillna('None')
all_data['MiscFeature'] = all_data['MiscFeature'].fillna('None')
all_data['Alley'] = all_data['Alley'].fillna('None')
all_data['Fence'] = all_data['Fence'].fillna('None')
all_data['FireplaceQu'] = all_data['FireplaceQu'].fillna('None')
all_data['LotFrontage'] = all_data.groupby('Neighborhood')['LotFrontage'].transform(lambda x: x.fillna(x.median()))

for col in ('GarageType', 'GarageFinish', 'GarageQual', 'GarageCond'):
    all_data[col] = all_data[col].fillna('None')
    
for col in ('GarageYrBlt', 'GarageArea', 'GarageCars'):
    all_data[col] = all_data[col].fillna(0)
    
for col in ('BsmtFinSF1', 'BsmtFinSF2', 'BsmtUnfSF','TotalBsmtSF', 'BsmtFullBath', 'BsmtHalfBath'):
    all_data[col] = all_data[col].fillna(0)
    
for col in ('BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2'):
    all_data[col] = all_data[col].fillna('None')
    
all_data['MasVnrType'] = all_data["MasVnrType"].fillna('None')
all_data['MasVnrArea'] = all_data["MasVnrArea"].fillna(0)
all_data['MSZoning'] = all_data['MSZoning'].fillna(all_data['MSZoning'].mode()[0])
all_data["Functional"] = all_data["Functional"].fillna("Typ")
all_data['Electrical'] = all_data['Electrical'].fillna(all_data['Electrical'].mode()[0])
all_data['KitchenQual'] = all_data['KitchenQual'].fillna(all_data['KitchenQual'].mode()[0])
all_data['Exterior1st'] = all_data['Exterior1st'].fillna(all_data['Exterior1st'].mode()[0])
all_data['Exterior2nd'] = all_data['Exterior2nd'].fillna(all_data['Exterior2nd'].mode()[0])
all_data['SaleType'] = all_data['SaleType'].fillna(all_data['SaleType'].mode()[0])
all_data['Utilities'] = all_data['Utilities'].fillna('None')

#Check remaining missing values if any 
all_data_na = (all_data.isnull().sum() / len(all_data)) * 100
all_data_na = all_data_na.drop(all_data_na[all_data_na == 0].index).sort_values(ascending=False)
missing_data = pd.DataFrame({'Missing Ratio' :all_data_na})
missing_data.head()

# transforming some numerical variables that are really categorical
all_data['MSSubClass'] = all_data['MSSubClass'].apply(str)
all_data['OverallCond'] = all_data['OverallCond'].astype(str)
all_data['YrSold'] = all_data['YrSold'].astype(str)
all_data['MoSold'] = all_data['MoSold'].astype(str)

cols = ('FireplaceQu', 'BsmtQual', 'BsmtCond', 'GarageQual', 'GarageCond', 'ExterQual', 
        'ExterCond','HeatingQC', 'PoolQC', 'KitchenQual', 'BsmtFinType1', 'BsmtFinType2', 
        'Functional', 'Fence', 'BsmtExposure', 'GarageFinish', 'LandSlope', 'LotShape', 
        'PavedDrive', 'Street', 'Alley', 'CentralAir', 'MSSubClass', 'OverallCond', 'YrSold', 'MoSold')
# process columns, apply LabelEncoder to categorical features
for c in cols:
    lbl = LabelEncoder() # Encode labels with value between 0 and n_classes-1.
    lbl.fit(list(all_data[c].values)) 
    all_data[c] = lbl.transform(list(all_data[c].values))

# shape        
print('Shape all_data: {}'.format(all_data.shape))
all_data.info()

# Adding total sqfootage feature 
all_data['TotalSF'] = all_data['TotalBsmtSF'] + all_data['1stFlrSF'] + all_data['2ndFlrSF']
all_data.shape

# Skew features
numeric_features = all_data.dtypes[all_data.dtypes != "object"].index
skewed_features = all_data[numeric_features].apply(lambda x: stats.skew(x.dropna())).sort_values(ascending=False)
skewness = pd.DataFrame({'Skew' :skewed_features})
print(skewness)

skewness = skewness[abs(skewness) > 0.75] 
skewed_features = skewness.index
lam = 0.15
for feat in skewed_features:
    all_data[feat] = boxcox1p(all_data[feat], lam) # Computes the Box-Cox transformation of 1 + x.
all_data[skewed_features]


# One-hot encode
all_data = pd.get_dummies(all_data)

train = all_data[:n_train]
test = all_data[n_train:]
train.iloc[0]


# Validation function
n_folds = 5

def rmsle_cv(model):
    kf = KFold(n_folds, shuffle = True, random_state = 1).get_n_splits(train.values)
    rmse = np.sqrt(-cross_val_score(model, train.values, y_train, scoring = 'neg_mean_squared_error', cv = kf))
    return rmse

# XGBoost
import xgboost as xgb
model_xgb = xgb.XGBRegressor(random_state = 1)
score = rmsle_cv(model_xgb)
score.mean()

model_xgb.fit(train, y_train)
xgb_pred = np.expm1(model_xgb.predict(test))

# Submission
submission = pd.DataFrame()
submission['Id'] = test_ID
submission['SalePrice'] = xgb_pred
submission.to_csv('../output/Housing_Price_Submission.csv', index=False)
