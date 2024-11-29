from flask import Flask, render_template, request, jsonify
import joblib
import pandas as pd
import io
import base64
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from flask_cors import CORS  # Add CORS support

app = Flask(__name__)
CORS(app)  # Enable CORS to allow requests from external clients (e.g., Power BI)

# Load the trained model and scaler
rf_model = joblib.load(r'model\rf.pkl')
dt_model = joblib.load(r'model\decision_tree_model.pkl')
le = LabelEncoder()

# Route for the home page
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        try:
            # Collect input values from the form
            pH = float(request.form['pH'])
            Nitrate = float(request.form['Nitrate'])
            Color = request.form['Color']
            Turbidity = float(request.form['Turbidity'])
            Odor = request.form['Odor']
            Chlorine = float(request.form['Chlorine'])
            Total_Dissolved_Solids = float(request.form['Total_Dissolved_Solids'])
            Water_Temperature = float(request.form['Water_Temperature'])

            

            # Create DataFrame for prediction
            input_data = pd.DataFrame({
                'pH': [pH],
                'Nitrate': [Nitrate],
                'Color': [Color],
                'Turbidity': [Turbidity],
                'Odor': [Odor],
                'Chlorine': [Chlorine],
                'Total Dissolved Solids': [Total_Dissolved_Solids],
                'Water Temperature': [Water_Temperature]
            })

            # Make predictions
            dt_prediction = dt_model.predict(input_data)
            rf_prediction = rf_model.predict(input_data)

            # Interpret results
            dt_result = 'Safe' if dt_prediction[0] == 1 else 'Unsafe'
            rf_result = 'Safe' if rf_prediction[0] == 1 else 'Unsafe'

            # Generate charts for each feature and comparison
            charts = {}
            features = input_data.columns
            expected_values = {
                'pH': 7.0,  # Example expected value
                'Nitrate': 10.0,  # Example expected threshold
                'Turbidity': 5.0,  # Example expected threshold
                'Chlorine': 1.0,  # Example expected threshold
                'Total Dissolved Solids': 500,  # Example expected threshold
                'Water Temperature': 25.0  # Example expected threshold
            }

            # Generate comparison charts
            for feature in features:
                fig, ax = plt.subplots(figsize=(4, 2))
                user_value = input_data[feature].iloc[0]
                expected_value = expected_values.get(feature, None)

                if expected_value is not None:
                    ax.bar([f"User Input ({feature})", f"Expected"], 
                           [user_value, expected_value], color=['blue', 'green'])
                else:
                    ax.bar(['User Input'], [user_value], color='blue')

                ax.set_title(f"Comparison for {feature}")
                ax.set_ylabel('Value')
                ax.grid(axis='y', linestyle='--', alpha=0.7)

                # Save the chart as a base64-encoded string
                img = io.BytesIO()
                fig.savefig(img, format='png', bbox_inches='tight')
                img.seek(0)
                charts[feature] = base64.b64encode(img.getvalue()).decode('utf8')

            # Return results and charts to the result page
            return render_template(
                'result.html',
                dt_prediction=dt_result,
                rf_prediction=rf_result,
                charts=charts,
                color=Color,  # Display original values for Color and Odor
                odor=Odor
            )
        except Exception as e:
            return f"Error: {str(e)}", 400
@app.route('/api/predict', methods=['POST'])
def api_predict():
    try:
        # Accept input as JSON
        data = request.json  

        # Extract features
        pH = float(data['pH'])
        Nitrate = float(data['Nitrate'])
        Color = data['Color']
        Turbidity = float(data['Turbidity'])
        Odor = data['Odor']
        Chlorine = float(data['Chlorine'])
        Total_Dissolved_Solids = float(data['Total_Dissolved_Solids'])
        Water_Temperature = float(data['Water_Temperature'])


        # Create DataFrame
        input_data = pd.DataFrame({
            'pH': [pH],
            'Nitrate': [Nitrate],
            'Color': [Color],
            'Turbidity': [Turbidity],
            'Odor': [Odor],
            'Chlorine': [Chlorine],
            'Total Dissolved Solids': [Total_Dissolved_Solids],
            'Water Temperature': [Water_Temperature]
        })

        # Predict with models
        dt_prediction = dt_model.predict(input_data)
        rf_prediction = rf_model.predict(input_data)

        # Format results
        dt_result = 'Safe' if dt_prediction[0] == 1 else 'Unsafe'
        rf_result = 'Safe' if rf_prediction[0] == 1 else 'Unsafe'

        # Return JSON response
        return jsonify({
            "decision_tree_result": dt_result,
            "random_forest_result": rf_result
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 400

# Run the Flask app
if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
