import joblib
import pandas as pd
from flask import Flask

app = Flask(__name__)

# Predict method for API call
@app.route('/predict', methods=['POST'])
def predict():
     json_ = request.json
     query_df = pd.DataFrame(json_)
     query = pd.get_dummies(query_df)
     prediction = clf.predict(query)
     return jsonify({'prediction': list(prediction)})

#  (1) load our persisted model into memory when the application starts
#  (2) create an endpoint that takes input variables, 
#  transforms them into the appropriate format, and returns predictions.
if __name__ == '__main__':
     clf = joblib.load('logistic_regression_model.pkl')
     app.run(port=8080)