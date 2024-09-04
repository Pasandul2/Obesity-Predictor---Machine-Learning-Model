from flask import Flask, request, render_template
import pickle
import numpy as np

# Initialize Flask app
app = Flask(__name__)

# Load the trained model
with open('rf_model.pkl', 'rb') as f:
    model = pickle.load(f)

# Load the scaler
with open('scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get form data
        Age = float(request.form['age'])
        Gender = int(request.form['gender'])  # Gender is now sent as an integer from the form
        Height = float(request.form['height'])
        Weight = float(request.form['weight'])
        BMI = float(request.form['bmi'])
        PhysicalActivityLevel = int(request.form['activity'])  # Physical Activity Level is now sent as an integer

        # Prepare input data
        input_data = np.array([[Age, Gender, Height, Weight, BMI, PhysicalActivityLevel]])
        input_data_scaled = scaler.transform(input_data)  # Apply the scaler

        # Predict
        prediction = model.predict(input_data_scaled)[0]

        # Map prediction to category
        category_map = {1: "Underweight", 2: "Normal weight", 3: "Overweight", 4: "Obese"}
        prediction_category = category_map.get(prediction, "Unknown category")

        return render_template('index.html', prediction_text=f'Obesity Category: {prediction_category}')
    
    except ValueError as ve:
        print(f"ValueError: {ve}")
        return render_template('index.html', prediction_text="Invalid input. Please check your inputs.")
    except Exception as e:
        print(f"Error: {e}")
        return render_template('index.html', prediction_text="An error occurred during prediction. Please try again.")
   
if __name__ == "__main__":
    app.run(debug=True)
