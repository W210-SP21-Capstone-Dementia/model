# Data-EDA-Model

## Data
`wget -r -np -nH --user=username --password=password https://media.talkbank.org/dementia/English/Pitt/` this contains the cookie test data

`wget -r -np -nH --user=username --password=password https://media.talkbank.org/dementia/0extra` this contains the ADReSS data

## Environment

Build the container X86
```
sudo docker build -t model_container -f dockerfile_model .

docker run --rm --name model_container -v ~/Documents/:/tf -p 8888:8888 -p 6006:6006 -e JUPYTER_ENABLE_LAB=yes --privileged -ti model_container:latest
```

Build the container ARM
```
sudo docker build -t model_container_arm -f dockerfile_model_arm .

docker run --rm --name model_container -v ~/Documents/:/tf -p 8888:8888 -p 6006:6006 -e JUPYTER_ENABLE_LAB=yes --privileged -ti model_container_arm:latest
```

## EDA

Please see `0-EDA+Baseline.ipynb` for EDA and base model built on spectrogram and convolutional nnet. 

## Baseline model
Baseline model is saved at `\saved_model\baseline`


## Serving

ARM Server
```
docker run --rm -p 8501:8501 -v /Users/zengm71/Documents/Berkeley/W210/W210-SP21-Capstone-Dementia/model/saved_model/base_line:/models/base_line -e MODEL_NAME=base_line -t emacski/tensorflow-serving:2.4.1 &
```
Client
```

curl -o -d '{"files": "/tf/Berkeley/W210/dementia/0extra/ADReSS-IS2020-data 2/train/Full_wave_enhanced_audio/cd/S129.wav"}' \
-X POST http://localhost:8501/v1/models/base_line:predict
```