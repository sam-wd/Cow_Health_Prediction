from flask import Flask, request, jsonify,render_template
import pickle
import numpy as np 

model_path = 'model.pkl'
with open(model_path, 'rb') as file:
    model = pickle.load(file)

app = Flask(__name__)



@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    float_features = [float(x) for x in request.form.values()]
    final_features = [np.array(float_features)]
    prediction = model.predict(final_features)
    output = 'Healthy' if prediction[0] == 1 else 'Unhealthy'
    return render_template('index.html', prediction_text='The prediction is : {}'.format(output))

if __name__ == "__main__":
    app.run(debug=True)
