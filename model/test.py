import joblib
import pandas as pd

# Load the trained model and scaler
model = joblib.load(r'C:\Users\Shruti\Desktop\Engineering\Final year project\GangaForecastingDSS\model\decision_tree_model.pkl')

input_data = pd.DataFrame({
    'pH': [7.0],  # Example values
    'Nitrate': [10.0],
    'Color': [1],  # Encoded values
    'Turbidity': [2.0],
    'Odor': [0],  # Encoded values
    'Chlorine': [0.5],
    'Total Dissolved Solids': [500.0],
    'Water Temperature': [25.0]
})
prediction = model.predict(input_data)
result = 'Safe' if prediction[0] == 1 else 'Unsafe'
print(result)
