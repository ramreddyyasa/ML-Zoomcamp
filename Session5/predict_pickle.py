import pickle

def predict_single(customer, dv, model):
    X = dv.transform([customer]) 
    y_pred = model.predict_proba(X)[:, 1]
    return y_pred[0]

with open('model1.bin', 'rb') as model_in:
    model = pickle.load(model_in)
model_in.close()

with open('dv.bin', 'rb') as dv_in:
    dv = pickle.load(dv_in)
dv_in.close()


customer = {"contract": "two_year", "tenure": 12, "monthlycharges": 19.7}

prediction = predict_single(customer, dv, model)
print("Score of the customer:", prediction)

