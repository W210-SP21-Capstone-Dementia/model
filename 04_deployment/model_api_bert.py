import joblib
import pandas as pd
import speech_recognition as sr
from sklearn.feature_extraction.text import TfidfVectorizer
from pydub import AudioSegment
from pydub.silence import split_on_silence 
from flask import Flask, request, jsonify, redirect
import os
import shutil
app = Flask(__name__)

r = sr.Recognizer()

# a function that splits the audio file into chunks 
# and applies speech recognition 
def silence_based_conversion(path, wav_file): 
    import os
    from datetime import datetime
    text_df = pd.DataFrame()

    input_file = path + wav_file
    filepath = input_file
    print(filepath)
    if not filepath.lower().endswith(".wav"):
        tmp_name = "tmp-bert"+datetime.now().strftime("%Y%m%d-%H%M%S")
        input_file = os.path.dirname(filepath)+"/"+tmp_name+".wav"
        cmd = "ffmpeg -i " + filepath + " " + input_file
        wav_file = tmp_name+".wav"
        os.system(cmd)

    # open the audio file stored in 
    # the local system as a wav file. 
    song = AudioSegment.from_wav(path + wav_file) 
    print(song)
  
    # open a file where we will concatenate   
    # and store the recognized text 
    text_file = wav_file.partition('.')[0] + ".txt"
    text_file_dir = path + "output_text/" 
    if not os.path.exists(text_file_dir):
        os.makedirs(text_file_dir)    
    text_file_path = text_file_dir + text_file
    fh = open(text_file_path, "w+") 
          

    dBFS = song.dBFS
    print('dBFS: ' + str(dBFS))
    # chunks = split_on_silence(song, 
    #     min_silence_len = 500,
    #     silence_thresh = dBFS-16,
    #     keep_silence = 250 
    # )

    # split track where silence is 0.5 seconds  
    # or more and get chunks 
    chunks = split_on_silence(song, 
        # must be silent for at least 0.5 seconds 
        # or 500 ms. adjust this value based on user 
        # requirement. if the speaker stays silent for  
        # longer, increase this value. else, decrease it. 
        min_silence_len = 250, 
  
        # consider it silent if quieter than -16 dBFS 
        # adjust this per requirement 
        silence_thresh = dBFS - 16
        # keep_silence = 250
    ) 
    # setting minimum length of each chunk to 25 seconds
    target_length = 20 * 1000 
    output_chunks = [chunks[0]]
    for chunk in chunks[1:]:
      print('Length of chunk: ' + str(len(output_chunks[-1])) )
      if len(output_chunks[-1]) < target_length:
        output_chunks[-1] += chunk
      else:
        # if the last output chunk is longer than the target length,
        # we can start a new one
        output_chunks.append(chunk)    
    # print(chunks)
  
    # create a directory to store the audio chunks. 
    try: 
        os.mkdir(path + 'bert_audio_chunks') 
    except(FileExistsError): 
        pass
  
    # move into the directory to 
    # store the audio files. 
    os.chdir(path +'bert_audio_chunks') 
  
    i = 0
    transcript = ''
    # process each chunk 
    for chunk in output_chunks: 
              
        # Create 0.5 seconds silence chunk 
        chunk_silent = AudioSegment.silent(duration = 10) 
  
        # add 0.5 sec silence to beginning and  
        # end of audio chunk. This is done so that 
        # it doesn't seem abruptly sliced. 
        audio_chunk = chunk_silent + chunk + chunk_silent 
  
        # export audio chunk and save it in  
        # the current directory. 
        text_file_id = text_file.partition('.')[0]
        chunk_file_name = text_file_id + "_" + "chunk" + str(i) + ".wav"
        print("saving " + chunk_file_name) 
        # specify the bitrate to be 192 k 
        audio_chunk.export("./" + chunk_file_name, bitrate ='192k', format ="wav") 
  
        # the name of the newly created chunk 
        filename = chunk_file_name
  
        print("Processing chunk file: " + filename) 
  
        # get the name of the newly created chunk 
        # in the AUDIO_FILE variable for later use. 
        file = filename 
  
        # create a speech recognition object 
        r = sr.Recognizer() 
  
        # recognize the chunk 
        with sr.AudioFile(file) as source: 
            # remove this if it is not working 
            # correctly. 
            #r.adjust_for_ambient_noise(source) 
            audio_listened = r.record(source) 
  
        try: 
            # try converting it to text 
            rec = r.recognize_google(audio_listened) 
            # write the output to the file. 
#             fh.write(rec+". ") 
#             text_df = text_df.append({'ID': text_file_id, 'Text': rec}, ignore_index = True)
            transcript = transcript + rec + ' '
        # catch any errors. 
        except sr.UnknownValueError: 
            print("Could not understand audio") 
  
        except sr.RequestError as e: 
            print("Could not request results. check your internet connection") 
  
        i += 1
    shutil.rmtree(path + 'bert_audio_chunks')     
    if not filepath.lower().endswith(".wav"):
        os.remove(input_file)
    return transcript

def model_serving_request(transcript, server_ip):

    import tensorflow as tf
    import numpy as np
    import json
    import requests	
    import os
    from datetime import datetime
    from transformers import BertTokenizer
    bert_name = 'bert-base-uncased'
    tokenizer = BertTokenizer.from_pretrained(bert_name,
                                            add_special_tokens=True,
                                            do_lower_case=True,
                                            max_length=256,
                                            pad_to_max_length=True)
    def bert_encoder(input_text):
        # txt = input_text.numpy().decode('utf-8')
        txt = input_text
        encoded = tokenizer.encode_plus(txt, add_special_tokens=True, 
                                        max_length=256, 
                                        pad_to_max_length=True, 
                                        return_attention_mask=True, 
                                        return_token_type_ids=True,
                                        truncation=True)
        return encoded['input_ids'], encoded['token_type_ids'], \
            encoded['attention_mask']

    bert_train = [bert_encoder(r) for r in transcript]
    bert_train = np.array(bert_train)
    sc_reviews, sc_segments, sc_masks = np.split(bert_train, 3, axis=1)

    data = json.dumps({
        "instances": sc_reviews.squeeze().reshape(1, 256).tolist()
        })
    headers = {"content-type": "application/json"}
    response = requests.post('http://' + server_ip + ':8501/v1/models/bert:predict', data=data, headers=headers, timeout=None)
    print(response)
    results = [x[0] for x in response.json()['predictions']]
    print(results)
    result = sum(results)/len(results)
    print(result)
   
    return result

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
    transcript = silence_based_conversion("/app/model/data/" , os.path.basename(file_path))      
    transcript = [transcript]
    print(transcript)

    model = data_payload['model']
    if model == 'base_model':
        score = model_serving_request(transcript, "model_server")
        data = {'dementia_score': score}
        return jsonify(data), 200
    else:
        return "not supported model!\n", 400

if __name__ == '__main__':
    app.run(debug=True,host='0.0.0.0')
