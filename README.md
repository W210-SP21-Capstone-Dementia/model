# Data-EDA-Model

## Data
`wget -r -np -nH --user=username --password=password https://media.talkbank.org/dementia/English/Pitt/` this contains the cookie test data

`wget -r -np -nH --user=username --password=password https://media.talkbank.org/dementia/0extra` this contains the ADReSS data

## Environment

Build the container X86
```
sudo docker build -t model_container -f dockerfile_model .

docker run --rm --name model_container -v ~/model/:/tf -p 8888:8888 -p 6006:6006 -e JUPYTER_ENABLE_LAB=yes --privileged -ti model_container:latest
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

### Local testing 
Create a bridge network
`docker network create --driver bridge tf_serving`

ARM Server
```
docker run --rm -p 8501:8501 -p 8500:8500 \
	--name model_server --network tf_serving \
	-v /Users/zengm71/Documents/Berkeley/W210/W210-SP21-Capstone-Dementia/model/saved_model/base_line:/models/base_line \
	-e MODEL_NAME=base_line \
	-t emacski/tensorflow-serving:2.4.1 &
```

x86 Server
```
docker run --rm -p 8501:8501 -p 8500:8500 \
	--name model_server --network tf_serving \
	-v ~/model/saved_model/base_line:/models/base_line \
	-e MODEL_NAME=base_line \
	-t tensorflow/serving:2.4.1 &
```

Client (if want to test the serving inside the container with a notebook)
```
docker run --rm --name model_container \
	--network tf_serving \
	-v ~/Documents/:/tf \
	-p 8888:8888 -p 6006:6006 \
	-e JUPYTER_ENABLE_LAB=yes \
	--privileged -ti model_container_arm:latest
```

Direct request
```
docker run --rm --name model_container \
	--network tf_serving \
	-v ~/Documents/Berkeley/W210/:/tf \
	--privileged \
	-ti model_container_arm:latest \
	bash /tf/W210-SP21-Capstone-Dementia/model/serving.sh
```

Remove network
`docker network rm tf_serving`

Build Flask Container
```
sudo docker build -t model_api -f dockerfile_model_api .
```

Start the docker container instance
```
docker run -d --name flask_api \
--network tf_serving \
-v /home/ubuntu/model/data:/model/data \
-p 5000:5000 \
model_api
```

Test API: (make sure the data S043.wav is under /model/data when you start the instance
```
curl -X POST -H "Content-Type: application/json" -d '{"file_path": "/home/ubuntu/model/data/S043.wav", "model": "base_model"}'  http://localhost:5000/getDementiaScore
```
