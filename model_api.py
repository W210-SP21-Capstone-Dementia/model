from flask import Flask, render_template,send_file, abort, jsonify, request
import json
import os
#to run "env FLASK_APP=model_api.py FLASK_ENV=development flask run"
app = Flask(__name__)

def model_serving_request(filepath, server_ip):

    import tensorflow as tf
    import numpy as np
    import json
    import requests	
    from datetime import datetime
    
    input_file = filepath
    if not filepath.lower().endswith(".wav"):
        tmp_name = "tmp-"+datetime.now().strftime("%Y%m%d-%H%M%S")
        input_file = os.path.dirname(filepath)+"/"+tmp_name+".wav"
        cmd = "ffmpeg -i " + filepath + " " + input_file
        os.system(cmd)
    
    audio_binary = tf.io.read_file(input_file)
    audio, _ = tf.audio.decode_wav(audio_binary)
    if audio.shape[1] > 1:
        audio = tf.reshape(audio[:, 0], (audio.shape[0],1))
        
    waveform = tf.squeeze(audio, axis=-1)    
    zero_padding = tf.zeros([10000000] - tf.shape(waveform), dtype=tf.float32)
    waveform = tf.cast(waveform, tf.float32)
    equal_length = tf.concat([waveform, zero_padding], 0)
    
    spectrogram = tf.signal.stft(equal_length, frame_length=255, frame_step=128)
    spectrogram = tf.abs(spectrogram)
    spectrogram = tf.expand_dims(spectrogram, -1)

    spectrogram = tf.expand_dims(spectrogram, axis = 0).numpy().tolist()
    data = json.dumps({
        "instances": spectrogram
        })
    headers = {"content-type": "application/json"}
    response = requests.post('http://' + server_ip + ':8501/v1/models/base_line:predict', data=data, headers=headers)
    result = response.json()['predictions'][0][0]
    print(result)
    if not filepath.lower().endswith(".wav"):
        os.remove(input_file)
    return result

@app.route("/")
def default_response():
    """the default response
    get:
        response:
            200:
                "This is the default response!"
            400:
                Not supported method


    """
    return "This is the default response!\n"


@app.route('/getDementiaScore',methods = ['POST'])
def getDementiaScore():
    """the getDementiaScore request handler
    post:
        parameter:
		    model: string (required)
		    file_path:string (required)
        response:
            200:
                data = {'dementia_score': int}
            400:
                Not supported method
    """
    if not request.is_json:
        return "Content not in JSON!\n",400
    data_payload = request.get_json()

    if data_payload is None or 'file_path' not in data_payload:
        return "Missing Input!\n", 400
    audio_path = data_payload['file_path']
    model = data_payload['model']

    if model == 'base_model':

        score = model_serving_request(audio_path, "model_server")
        data = {'dementia_score': score}
        return jsonify(data), 200
    else:
        return "not supported model!\n", 400

if __name__ == '__main__':
    app.run(debug=True,host='0.0.0.0')
