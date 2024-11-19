from flask import Flask, render_template, request, redirect, url_for
import joblib
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Use Agg backend for non-GUI operations
import io
import base64
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, LabelEncoder

app = Flask(__name__)

# Load the trained model
model = joblib.load(r'model\decision_tree_model.pkl')

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

        # Preprocess the data (encoding Color and Odor)
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

        # For debugging, comment out the scaling
        # numerical_cols = ['pH', 'Nitrate', 'Turbidity', 'Chlorine', 'Total Dissolved Solids', 'Water Temperature']
        # input_data[numerical_cols] = scaler.fit_transform(input_data[numerical_cols])

        # Make prediction
        prediction = model.predict(input_data)
        result = 'Safe' if prediction[0] == 1 else 'Unsafe'

        fig, ax = plt.subplots(figsize=(12, 8))  # Increase figure size for better label spacing
        ax.bar(input_data.columns, input_data.iloc[0].values, color='skyblue')

        # Rotate the x-axis labels and align them to the right
        plt.xticks(rotation=45, ha='right')

        ax.set_xlabel('Parameters')
        ax.set_ylabel('Values')
        ax.set_title('Water Quality Parameters')

        # Convert the plot to a PNG image in memory
        img = io.BytesIO()
        fig.savefig(img, format='png')
        img.seek(0)
        img_data = base64.b64encode(img.getvalue()).decode('utf-8')

        return render_template('result.html', prediction=result, img_data=img_data)


# Run the Flask app
if __name__ == '__main__':
    app.run(debug=True)
