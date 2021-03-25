import joblib
import pandas as pd
import speech_recognition as sr
from sklearn.feature_extraction.text import TfidfVectorizer
from flask import Flask, request, jsonify, redirect
import os

app = Flask(__name__)

# These will be populated at training time
model_columns = None
clf = None

# Predict method for API call
@app.route('/getDementiaScore', methods=['POST'])
def getDementiaScore():
    # Transcription from https://blog.thecodex.me/speech-recognition-with-python-and-flask/

    if not request.is_json:
        return "Content not in JSON!\n",400
    data_payload = request.get_json()

    if data_payload is None or 'file_path' not in data_payload:
        return "Missing Input!\n", 400

    file_path = data_payload['file_path']
            
    audio_path = "/app/model/data/" + os.path.basename(file_path)
    recognizer = sr.Recognizer()
    audioFile = sr.AudioFile(audio_path)
    with audioFile as source:
        data = recognizer.record(source)
    transcript = recognizer.recognize_google(data, key=None)
    transcript = [transcript]
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

            prediction = float(clf.predict(query))

            # Converting to int from int64
            return_obj = jsonify({"prediction": prediction})
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
        clf = joblib.load('/app/model/sklearn-text-based/logistic_regression_model.pkl')
        print('logistic regression model loaded')

        # Also we have to load model columns when the application starts.
        model_columns = joblib.load('/app/model/sklearn-text-based/logistic_regression_model_columns.pkl')
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
    app.run(debug=True,host='0.0.0.0')
