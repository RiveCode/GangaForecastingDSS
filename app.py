from flask import Flask, render_template, request, jsonify
import joblib
import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder

app = Flask(__name__)

# Load the trained model
model = joblib.load(r'C:\Users\Shruti\Desktop\Engineering\Final year project\GangaForecastingDSS\model\decision_tree_model.pkl')

# Label Encoder and StandardScaler used during training
scaler = StandardScaler()
le = LabelEncoder()

# Route for the home page
@app.route('/')
def index():
    return render_template('index.html')

# Route to make predictions
@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        # Get user input data
        pH = float(request.form['pH'])
        Nitrate = float(request.form['Nitrate'])
        Color = request.form['Color']
        Turbidity = float(request.form['Turbidity'])
        Odor = request.form['Odor']
        Chlorine = float(request.form['Chlorine'])
        Total_Dissolved_Solids = float(request.form['Total_Dissolved_Solids'])
        Water_Temperature = float(request.form['Water_Temperature'])

        # Preprocess the data (encoding Color and Odor, scaling numerical values)
        color_encoded = le.fit_transform([Color])[0]
        odor_encoded = le.fit_transform([Odor])[0]
        
        # Create a DataFrame for the input
        input_data = pd.DataFrame({
            'pH': [pH],
            'Nitrate': [Nitrate],
            'Color': [color_encoded],
            'Turbidity': [Turbidity],
            'Odor': [odor_encoded],
            'Chlorine': [Chlorine],
            'Total Dissolved Solids': [Total_Dissolved_Solids],
            'Water Temperature': [Water_Temperature]
        })

        # Standardize numerical columns
        numerical_cols = ['pH', 'Nitrate', 'Turbidity', 'Chlorine', 
                          'Total Dissolved Solids', 'Water Temperature']
        input_data[numerical_cols] = scaler.fit_transform(input_data[numerical_cols])

        # Make prediction
        prediction = model.predict(input_data)
        result = 'Safe' if prediction[0] == 1 else 'Unsafe'

        return render_template('result.html', prediction=result)

# Run the Flask app
if __name__ == '__main__':
    app.run(debug=True)
