def startConversion(path, filename, lang = lang_):
                
    full_path = path + filename

    with sr.AudioFile(full_path) as source:
        print('Transcribing file: ' + str(filename) + ' in path: ' + str(full_path))
        audio_text = r.listen(source)
        # recognize_() method will throw a request error if the API is unreachable, hence using exception handling
        try:

            # using google speech recognition
            # print('Converting audio transcripts into text ...')
            text = r.recognize_google(audio_text)

            return text
            
        except Exception as e:
            print('Error: ' + str(e) + ' <- this guy')

def model_serving_request(filepath, server_ip):

    import pandas as pd
    import numpy as np
    import speech_recognition as sr
    from sklearn.ensemble import RandomForestClassifier, StackingClassifier
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.feature_extraction.text import CountVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import classification_report, accuracy_score, confusion_matrix, plot_confusion_matrix
    from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.neural_network import MLPClassifier
    import json
    import requests	
    import os
    from datetime import datetime
    
    input_file = filepath
    filename = filepath.split(sep='/')[-1][0]
    justfilepath = filepath.split(sep='/')[:-1][0]

    text_path = justfilepath + '/transcription/'
    lang_ = 'en-US'

    if not filepath.lower().endswith(".wav"):
        tmp_name = "tmp-"+datetime.now().strftime("%Y%m%d-%H%M%S")
        input_file = os.path.dirname(filepath)+"/"+tmp_name+".wav"
        cmd = "ffmpeg -i " + filepath + " " + input_file
        os.system(cmd)
    
    text_dict = {"ID": [], "Text": []}

    data = startConversion(path = justfilepath, filename = filename)
    just_name = filename.split(sep='.')[:-1][0]
    text_dict["ID"].append(just_name)
    text_dict["Text"].append(data)

    audio_binary = tf.io.read_file(input_file)
    audio, _ = tf.audio.decode_wav(audio_binary)
    if audio.shape[1] > 1:
        audio = tf.reshape(audio[:, 0], (audio.shape[0],1))
        
    waveform = tf.squeeze(audio, axis=-1)    
    rolling_waveform_tensors = window(waveform, size=_*30, stride=_*1)
    rolling_spectrograms = tf.signal.stft(rolling_waveform_tensors, frame_length=512, frame_step=_)
    rolling_spectrograms = tf.abs(rolling_spectrograms)
    rolling_spectrograms = tf.expand_dims(rolling_spectrograms, -1)
    rolling_spectrograms = rolling_spectrograms.numpy().tolist()
    data = json.dumps({
        "instances": rolling_spectrograms
        })
    headers = {"content-type": "application/json"}
    response = requests.post('http://' + server_ip + ':8501/v1/models/base_line:predict', data=data, headers=headers)
    results = [x[0] for x in response.json()['predictions']]
    result = sum(results)/len(results)
    print(result)
    if not filepath.lower().endswith(".wav"):
        os.remove(input_file)
    return result

if __name__ == "__main__":
    # execute only if run as a script
    import sys
    model_serving_request(sys.argv[1], sys.argv[2])