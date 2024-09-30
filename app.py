from flask import Flask, request, jsonify, render_template
import pickle
from flask_caching import Cache

app = Flask(__name__)

# Load the model and dataset
with open('government_scheme_model.pkl', 'rb') as f:
    model, df = pickle.load(f)

# Function to get scheme information
def get_scheme_info(scheme_name):
    # Predict the closest matching scheme
    predicted_scheme = model.predict([scheme_name])[0]
    
    # Find the corresponding row in the dataset
    scheme = df[df['Scheme Name'] == predicted_scheme].iloc[0]
    
    

    
    # Return scheme details
    return {
        'Scheme Name': scheme['Scheme Name'],
        'Genre': scheme['Genre'],
        'Description': scheme['Description'],
        'Eligibility Criteria': f"Applicable Age: {scheme['Applicable Age']}, Gender: {scheme['Gender']}, Income Range: {scheme['Income Range']}"
    }



@app.route('/')
def index():
    return render_template('index.html')

@app.route('/get_scheme_info', methods=['POST'])
def get_scheme():
    scheme_name = request.form['scheme_name']
    scheme_info = get_scheme_info(scheme_name)
    return jsonify(scheme_info)

if __name__ == '__main__':
    app.run(debug=True , port=8000)
