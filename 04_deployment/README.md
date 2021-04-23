# Model Environment

Build the container X86
```
sudo docker build -t model_container -f dockerfile_model .

docker run --rm --network tf_serving --name model_container -v ~/model/:/tf -p 8888:8888 -p 6006:6006 -e JUPYTER_ENABLE_LAB=yes --privileged -ti model_container:latest
```

Build the container ARM
```
sudo docker build -t model_container_arm -f dockerfile_model_arm .

docker run --rm --network tf_serving --name model_container -v ~/Documents/:/tf -p 8888:8888 -p 6006:6006 -e JUPYTER_ENABLE_LAB=yes --privileged -ti model_container_arm:latest
```

# Serving

All services are containerized using docker. We used the deep learning AMI from AWS with Ubuntu 16.04. 

## Network
Create a bridge network
`docker network create --driver bridge tf_serving`

Remove network
`docker network rm tf_serving`

## Deploy multiple models
* tf_serving, see `model_config.config` for setup
	```
	docker run --rm -p 8501:8501 -p 8500:8500 \
	--name model_server \
	--network tf_serving \
	-v ~/model/saved_model/base_line:/models/base_line \
	-v ~/model/saved_model/michael/rl_lstm_wn_0314:/models/lstm/1/ \
	-v ~/model/saved_model/michael/eGeMAPS:/models/smile/1/ \
	-v ~/model/saved_model/michael/BERT:/models/bert/1/ \
	-v ~/model/model_config.config:/models/model_config.config \
	-t tensorflow/serving:2.4.1 \
	--model_config_file=/models/model_config.config 
	```

### Baseline
* flask 
	`docker build -t model_api -f dockerfile_model_api .`
	`docker run --rm -d --name flask_api --network tf_serving -v /home/ubuntu/model:/app/model -p 5000:5000 model_api`

* test `curl -X POST -H "Content-Type: application/json" -d '{"file_path": "/home/ubuntu/model/data/S041.wav", "model": "base_model"}'  http://localhost:5000/getDementiaScore` expect score 27.956016672093025

### LSTM + white noise
* flask 
	`docker build -t model_api_lstm -f dockerfile_model_api_lstm .`
	`docker run --rm -d --name flask_api_lstm --network tf_serving -v /home/ubuntu/model:/app/model -p 5001:5000 model_api_lstm`

* test `curl -X POST -H "Content-Type: application/json" -d '{"file_path": "/home/ubuntu/model/data/S041.wav", "model": "base_model"}'  http://localhost:5001/getDementiaScore` expect score 26.911125

### Smile 
* flask 
	`docker build -t model_api_smile -f dockerfile_model_api_smile .`
	`docker run --rm -d --name flask_api_smile --network tf_serving -v /home/ubuntu/model:/app/model -p 5002:5000 model_api_smile`

* test `curl -X POST -H "Content-Type: application/json" -d '{"file_path": "/home/ubuntu/model/data/S041.wav", "model": "base_model"}'  http://localhost:5002/getDementiaScore` expect score 25.18345991930233

### Text Model
* flask
	`docker build -t model_api_text -f dockerfile_model_api_text .`
	`docker run --rm -d --network tf_serving --name flask_api_text -v /home/ubuntu/model:/app/model -p 5003:5000 model_api_text`
* test 
	`curl -X POST -H "Content-Type: application/json" -d '{"file_path": "/home/ubuntu/model/data/S041.wav"}' http://localhost:5003/getDementiaScore`

## Text BERT Model
* flask
	`docker build -t model_api_text_bert -f dockerfile_model_api_text_bert .`
	`docker run --network tf_serving --rm -d --name flask_api_text_bert -v /home/ubuntu/model:/app/model -p 5004:5000 model_api_text_bert`
* test 
	`curl -X POST -H "Content-Type: application/json" -d '{"file_path": "/home/ubuntu/model/data/S041.wav", "model": "base_model"}' http://localhost:5004/getDementiaScore` expect score 25.4926872

## Local testing
* ARM Server
```
docker run --rm -p 8501:8501 -p 8500:8500 \
	--name model_server --network tf_serving \
	-v /Users/zengm71/Documents/Berkeley/W210/W210-SP21-Capstone-Dementia/model/saved_model/base_line:/models/base_line \
	-e MODEL_NAME=base_line \
	-t emacski/tensorflow-serving:2.4.1 &
```

* x86 Server
```
docker run --rm -p 8501:8501 -p 8500:8500 \
	--name model_server --network tf_serving \
	-v ~/model/saved_model/base_line:/models/base_line \
	-e MODEL_NAME=base_line \
	-t tensorflow/serving:2.4.1 &
```

* Client (if want to test the serving inside the container with a notebook)
```
docker run --rm --name model_container \
	--network tf_serving \
	-v ~/Documents/:/tf \
	-p 8888:8888 -p 6006:6006 \
	-e JUPYTER_ENABLE_LAB=yes \
	--privileged -ti model_container_arm:latest
```

* Direct request
```
docker run --rm --name model_container \
	--network tf_serving \
	-v ~/model:/tf \
	--privileged \
	-ti model_container:latest \
	bash /tf/serving.sh
```

* Request through Flask
```
sudo docker build -t model_api -f dockerfile_model_api .
```

	Start the docker container instance
	```
	docker run -d --name flask_api --network tf_serving \
	-v /home/ubuntu/model:/app/model \
	-p 5000:5000 model_api
	```

	Test API: (make sure the data S043.wav is under /model/data when you start the instance
	```
	curl -X POST -H "Content-Type: application/json" -d '{"file_path": "/home/ubuntu/model/data/S043.wav", "model": "base_model"}'  http://localhost:5000/getDementiaScore
```