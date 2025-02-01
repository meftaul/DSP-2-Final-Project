# TODO: Orgranize the code into functions

import pandas as pd
import numpy as np

# load data
data = pd.read_csv('data/housing.csv')

print(data.head())

data['income_cat'] = pd.cut(data['median_income'], 
                            bins=[0., 1.5, 3.0, 4.5, 6., np.inf], 
                            labels=[1, 2, 3, 4, 5])

# split data
from sklearn.model_selection import train_test_split

train_set, test_set = train_test_split(data, test_size=0.2, random_state=42, stratify=data['income_cat'])

# drop income_cat
train_set.drop('income_cat', axis=1, inplace=True)
test_set.drop('income_cat', axis=1, inplace=True)

print(f'Train set shape: {train_set.shape}', f'Test set shape: {test_set.shape}')

# save train and test set
import os

os.makedirs('data', exist_ok=True)

train_set.to_csv('data/train.csv', index=False)
test_set.to_csv('data/test.csv', index=False)

# ----------------------------

train_set = pd.read_csv('data/train.csv')

# split features and target
X_train = train_set.drop('median_house_value', axis=1)
y_train = train_set['median_house_value'].copy()

# validation set
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, 
                                                  test_size=0.2, 
                                                  random_state=42)

print(f'Train set shape: {X_train.shape}', f'Validation set shape: {X_val.shape}')
print(f'Train target shape: {y_train.shape}', f'Validation target shape: {y_val.shape}')

# numerical and categorical columns
num_cols = X_train.select_dtypes(include=[np.number]).columns.tolist()
cat_cols = X_train.select_dtypes('object').columns.tolist()

print(f'Numerical columns: {num_cols}', f'Categorical columns: {cat_cols}')

# import SimpleImputer, StandardScaler, OrdinalEncoder

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OrdinalEncoder

num_imputer = SimpleImputer(strategy='mean')
scaler = StandardScaler()

# apply imputer and scaler for numerical columns
X_train[num_cols] = num_imputer.fit_transform(X_train[num_cols])
X_train[num_cols] = scaler.fit_transform(X_train[num_cols])

X_val[num_cols] = num_imputer.transform(X_val[num_cols])
X_val[num_cols] = scaler.transform(X_val[num_cols])

# for categorical columns
cat_imputer = SimpleImputer(strategy='most_frequent')
encoder = OrdinalEncoder()

X_train[cat_cols] = cat_imputer.fit_transform(X_train[cat_cols])
X_train[cat_cols] = encoder.fit_transform(X_train[cat_cols])

X_val[cat_cols] = cat_imputer.transform(X_val[cat_cols])
X_val[cat_cols] = encoder.transform(X_val[cat_cols])


print(X_train.shape, X_val.shape)
print(y_train.shape, y_val.shape)

# create the model
from sklearn.linear_model import LinearRegression

lin_reg = LinearRegression()

# fit the model
lin_reg.fit(X_train, y_train)

# predict on validation set
y_pred = lin_reg.predict(X_val)

# evaluate the model
from sklearn.metrics import root_mean_squared_error

rmse = root_mean_squared_error(y_val, y_pred)

print(f'RMSE: {rmse}')

# create a random forest model
from sklearn.ensemble import RandomForestRegressor

rf = RandomForestRegressor(n_estimators=120, random_state=42)

# fit the model
rf.fit(X_train, y_train)

# predict on validation set
y_pred = rf.predict(X_val)

# evaluate the model
rmse = root_mean_squared_error(y_val, y_pred)

print(f'RMSE: {rmse}')

# save the model
import joblib

os.makedirs('models', exist_ok=True)

joblib.dump(rf, 'models/rf_model.pkl')
joblib.dump(scaler, 'models/scaler.pkl')
joblib.dump(encoder, 'models/encoder.pkl')
joblib.dump(num_imputer, 'models/num_imputer.pkl')
joblib.dump(cat_imputer, 'models/cat_imputer.pkl')