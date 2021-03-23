import joblib
import pandas as pd
import speech_recognition as sr
from sklearn.feature_extraction.text import TfidfVectorizer
from flask import Flask, request, jsonify, redirect

app = Flask(__name__)

# These will be populated at training time
model_columns = None
clf = None

# Predict method for API call
@app.route('/predict', methods=['POST'])
def predict():
    # Transcription from https://blog.thecodex.me/speech-recognition-with-python-and-flask/
    transcript = ""
    if request.method == "POST":
        print("FORM DATA RECEIVED")

        if "file" not in request.files:
            return redirect(request.url)

        file = request.files["file"]
        if file.filename == "":
            return redirect(request.url)
            
        if file:
            recognizer = sr.Recognizer()
            audioFile = sr.AudioFile(file)
            with audioFile as source:
                data = recognizer.record(source)
            transcript = recognizer.recognize_google(data, key=None)

    if clf:
        try:
            # json_ = request.json
            # query_df = pd.DataFrame(json_)
            # query = pd.get_dummies(query_df)
            vectorizer = TfidfVectorizer()
            vectorized_text = vectorizer.fit_transform(transcript)
            vectorizedTextDF = pd.DataFrame(vectorized_text.toarray(), columns=vectorizer.get_feature_names())                        

            query = pd.get_dummies(vectorizedTextDF)

            # We have the list of columns persisted, 
            # so we can just replace the missing values with zeros 
            # at the time of prediction.
            query = query.reindex(columns=model_columns, fill_value=0)

            prediction = list(clf.predict(query))

            # Converting to int from int64
            return_obj = jsonify({"prediction": list(map(int, prediction))})
            print(return_obj)
            return return_obj
        except Exception as e:

            # return jsonify({'error': str(e), 'trace': traceback.format_exc()})
            return_obj_e = jsonify({'error': str(e)})
            print(return_obj_e)
            return return_obj_e
    else:
        print('train first')
        return 'no model here'

#  (1) load our persisted model into memory when the application starts
#  (2) create an endpoint that takes input variables, 
#  transforms them into the appropriate format, and returns predictions.
if __name__ == '__main__':
    try:
        # Load persisted model
        clf = joblib.load('logistic_regression_model.pkl')
        print('logistic regression model loaded')

        # Also we have to load model columns when the application starts.
        model_columns = joblib.load('logistic_regression_model_columns.pkl')
        print('logistic regression model columns loaded')

        ### BEGIN TESTING FRAMEWORK
        # test_cc_filepath = './dementia/0extra/ADReSS-IS2020-train/ADReSS-IS2020-data/train/Full_wave_enhanced_audio/cc/S001.wav'
        # app.post('/predict', data=test_cc_filepath)
        ### END TESTING FRAMEWORK
    except Exception as e:
        print('Either model or columns are missing')
        print(str(e))
        clf = None

    # TODO - change port and host number
    app.run(host='localhost', port=8080, debug=True)