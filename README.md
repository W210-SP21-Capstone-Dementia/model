# model
Data-EDA-Model

## Data
`wget -r -np -nH --user=username --password=password https://media.talkbank.org/dementia/English/Pitt/` this contains the cookie test data

`wget -r -np -nH --user=username --password=password https://media.talkbank.org/dementia/0extra` this contains the ADReSS data

## Environment

Build the container
```
sudo docker build -t model_container -f dockerfile_model .

docker run --rm --name model_container -v ~/Documents/:/tf -p 8888:8888 -p 6006:6006 -e JUPYTER_ENABLE_LAB=yes --privileged -ti model_container:latest
```


## EDA
