import pickle
from flask import Flask, request, jsonify

def predict_single(customer, dv, model):
    X = dv.transform([customer]) 
    y_pred = model.predict_proba(X)[:, 1]
    return y_pred[0]

with open('model2.bin', 'rb') as model_in:
    model = pickle.load(model_in)
model_in.close()

with open('dv.bin', 'rb') as dv_in:
    dv = pickle.load(dv_in)
dv_in.close()

app = Flask('churn')


@app.route('/predict', methods=['POST'])
def predict():
    customer = request.get_json()

    prediction = predict_single(customer, dv, model)
    churn = prediction >= 0.5
    
    result = {
        'churn_probability': float(prediction),
        'churn': bool(churn),
    }

    return jsonify(result)


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=9696)
