import joblib
import pandas as pd
sample_data = {
    'longitude': -124.23, 
    'latitude': 40.54, 
    'housing_median_age': 52.0, 
    'total_rooms': 2694.0, 
    'total_bedrooms': 453.0, 
    'population': 1152.0, 
    'households': 435.0, 
    'median_income': 3.0806, 
    'median_house_value': 106700.0, 
    'ocean_proximity': 'NEAR OCEAN'
}
sample_data_df = pd.DataFrame([sample_data])
model = joblib.load('models/model_with_pipeline.pkl')
result = model.predict(sample_data_df)
print(result[0])