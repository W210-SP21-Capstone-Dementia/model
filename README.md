# model
Data-EDA-Model

## Data
`wget -r -np -nH --user=username --password=password https://media.talkbank.org/dementia/English/Pitt/` this contains the cookie test data

`wget -r -np -nH --user=username --password=password https://media.talkbank.org/dementia/0extra` this contains the ADReSS data

## Environment (we will set up a docker image later)

```
conda create -n audio python=3.7
conda activate audio
pip install -r requirements.txt
```

```
ipython kernel install --user --name=audio
conda activate audio
jupyter-notebook
```
## EDA