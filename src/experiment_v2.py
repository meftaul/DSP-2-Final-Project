# import pandas as pd

# # load data
# test_data = pd.read_csv('data/test.csv')

# print(test_data.head())

# X_test = test_data.drop('median_house_value', axis=1)
# y_test = test_data['median_house_value'].copy()

# print(f'Test set shape: {X_test.shape}', f'Test target shape: {y_test.shape}')

# # numerical and categorical columns
# num_cols = X_test.select_dtypes(include=[np.number]).columns.tolist()
# cat_cols = X_test.select_dtypes('object').columns.tolist()

# print(f'Numerical columns: {num_cols}', f'Categorical columns: {cat_cols}')

# import joblib

# rf = joblib.load('models/rf_model.pkl')

# scaler = joblib.load('models/scaler.pkl')
# encoder = joblib.load('models/encoder.pkl')
# num_imputer = joblib.load('models/num_imputer.pkl')
# cat_imputer = joblib.load('models/cat_imputer.pkl')

# X_test[num_cols] = num_imputer.transform(X_test[num_cols])
# X_test[num_cols] = scaler.transform(X_test[num_cols])

# X_test[cat_cols] = cat_imputer.transform(X_test[cat_cols])
# X_test[cat_cols] = encoder.transform(X_test[cat_cols])

# print(X_test.shape, y_test.shape)

# # make predictions
# rf_preds = rf.predict(X_test)

# # evaluate the model
# from sklearn.metrics import root_mean_squared_error

# rmse = root_mean_squared_error(y_test, rf_preds)

# print(f'Root Mean Squared Error: {rmse}')

import joblib
import pandas as pd

# load the model
model = joblib.load('models/model_with_pipeline.pkl')

test_data = pd.read_csv('data/test.csv')
X_test = test_data.drop('median_house_value', axis=1)
y_test = test_data['median_house_value'].copy()

# make predictions
y_preds = model.predict(X_test)
# evaluate the model
from sklearn.metrics import root_mean_squared_error
rmse = root_mean_squared_error(y_test, y_preds)
print(f'Root Mean Squared Error: {rmse}')


test_data.sample(1).to_dict()