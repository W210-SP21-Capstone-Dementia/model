from flask import Flask, render_template,send_file, abort, jsonify, request
import json
import os
#to run "env FLASK_APP=model_api.py FLASK_ENV=development flask run"
app = Flask(__name__)

def window(x, size, stride):
    import tensorflow as tf
    length = int(len(x))
    if length // size == 0:
        zero_padding =  tf.zeros([size] - tf.shape(x), dtype=tf.float32)
        x = tf.cast(x, tf.float32)
        x = tf.concat([x, zero_padding], 0)
        length = int(len(x))
    return tf.map_fn(lambda i: x[i*stride:i*stride+size], tf.range((length-size)//stride+1), dtype=tf.float32)
    
def model_serving_request(filepath, server_ip):

    import tensorflow as tf
    import numpy as np
    import json
    import requests	
    import os
    from datetime import datetime, timedelta
    import opensmile
    from pydub import AudioSegment
    import glob
    import shutil

    input_file = filepath
    if not filepath.lower().endswith(".wav"):
        tmp_name = "tmp-smile"+datetime.now().strftime("%Y%m%d-%H%M%S")
        input_file = os.path.dirname(filepath)+"/"+tmp_name+".wav"
        cmd = "ffmpeg -i " + filepath + " " + input_file
        os.system(cmd)

    newAudio = AudioSegment.from_wav(input_file)
    newAudio = newAudio.set_channels(1)
    if len(newAudio)/1000 < 30:
            newAudio = newAudio + AudioSegment.silent(duration=30000- len(newAudio))

    smile = opensmile.Smile(feature_set=opensmile.FeatureSet.eGeMAPSv01b, num_workers=8)
    os.mkdir('/app/model/data/temp_smile/')
    for t1 in list(range(int(len(newAudio)/1000-29))):
        exportAudio = newAudio[t1*1000:(t1 + 30)*1000]
        exportAudio.export('/app/model/data/temp_smile/temp' + str(t1) + '.wav' , format="wav")
        
    to_predict = smile.process_files(glob.glob('/app/model/data/temp_smile/*.wav'), channel = 0)
    # newAudio.export('/app/model/data/temp_smile/export.wav' , format="wav")
    # for t1 in list(range(int(len(newAudio)/1000-29))):
    #     if t1 == 0:
    #         to_predict = smile.process_file('/app/model/data/temp_smile/export.wav', 
    #                                         start = timedelta(seconds=0), 
    #                                         end = timedelta(seconds=30), 
    #                                         channel=0)
    #     else: 
    #         to_predict = to_predict.append(smile.process_file('/app/model/data/temp_smile/export.wav', 
    #                                         start = timedelta(seconds=t1), 
    #                                         end = timedelta(seconds=t1+30), 
    #                                         channel=0))
    to_predict = to_predict.iloc[:, 0:88].to_numpy()
    
    shutil.rmtree('/app/model/data/temp_smile')     

    data = json.dumps({
        "instances": to_predict.tolist()
        })
    headers = {"content-type": "application/json"}
    response = requests.post('http://' + server_ip + ':8501/v1/models/smile:predict', data=data, headers=headers)
    results = [x[0] for x in response.json()['predictions']]
    result = sum(results)/len(results) * 30
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
    file_path = data_payload['file_path']
    audio_path = "/app/model/data/" + os.path.basename(file_path)
    model = data_payload['model']

    if model == 'base_model':

        score = model_serving_request(audio_path, "model_server")
        data = {'dementia_score': score}
        return jsonify(data), 200
    else:
        return "not supported model!\n", 400

if __name__ == '__main__':
    app.run(debug=True,host='0.0.0.0')
