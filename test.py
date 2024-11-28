import requests

# Define the Flask API endpoint URL
url = "http://127.0.0.1:5000/api/predict"

# Define the input data to send to the API
input_data = {
    "pH": 7.0,  # Example values
    "Nitrate": 10.0,
    "Color": "Clear",  # Use raw values (not encoded); the Flask API handles encoding
    "Turbidity": 2.0,
    "Odor": "None",  # Use raw values
    "Chlorine": 0.5,
    "Total_Dissolved_Solids": 500.0,
    "Water_Temperature": 25.0
}

try:
    # Send a POST request to the Flask API
    response = requests.post(url, json=input_data)

    # Check if the request was successful
    if response.status_code == 200:
        # Parse and print the API response
        result = response.json()
        print("Decision Tree Result:", result["decision_tree_result"])
        print("Random Forest Result:", result["random_forest_result"])
    else:
        print("Failed to connect to the API. Status Code:", response.status_code)
        print("Error message:", response.text)

except requests.exceptions.RequestException as e:
    print(f"An error occurred: {e}")
