import pandas as pd
from sklearn.cross_validation import train_test_split
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import cross_val_score
from xgboost import XGBRegressor


# READ TRAINING FILE
ames_data = pd.read_csv('input/train.csv')
ames_features = ['LotArea', 'OverallQual', 'OverallCond', 'FullBath', 'HalfBath', 'BedroomAbvGr', 
					'YearBuilt', 'YrSold']


# PREPROCESSING
X = ames_data[ames_features]
y = ames_data.SalePrice

X_train, X_val, y_train, y_val = train_test_split(X, y, random_state=1)


# DEFINING MODEL AND FIT
ames_model = XGBRegressor(random_state=1).fit(X_train,y_train)


# ERROR CALCULATION
predicted_home_prices = ames_model.predict(X_train)
MAE = mean_absolute_error(y_train, predicted_home_prices)
scores = cross_val_score(ames_model, X, y, scoring='neg_mean_absolute_error')


# PREDICTIONS FROM TEST DATA
test = pd.read_csv('input/test.csv')
test_X = test[ames_features]
#test_X.fillna(0, inplace=True)

predicted_prices = ames_model.predict(test_X)

# OUTPUT TO TERMINAL
print('MAE: {}'.format(MAE))
print('Cross Val Score: {}'.format(scores))


# OUTPUT TO CSV
submission = pd.DataFrame({'Id': test.Id, 'SalePrice': predicted_prices})
submission.to_csv('output/XGB_submission.csv', index=False)
