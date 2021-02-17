def model_serving_request(filepath, server_ip):

    import tensorflow as tf
    import numpy as np
    import json
    import requests	
    import os
    from datetime import datetime
    
    input_file = filepath
    if not filepath.lower().endswith(".wav"):
        tmp_name = "tmp-"+datetime.now().strftime("%Y%m%d-%H%M%S")
        input_file = os.path.dirname(filepath)+"/"+tmp_name+".wav"
        cmd = "ffmpeg -i " + filepath + " " + input_file
        os.system(cmd)
    
    audio_binary = tf.io.read_file(input_file)
    audio, _ = tf.audio.decode_wav(audio_binary)

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
    return result

if __name__ == "__main__":
    # execute only if run as a script
    import sys
    model_serving_request(sys.argv[1], sys.argv[2])
