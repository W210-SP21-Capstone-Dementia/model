* `0-EDA+Baseline.ipynb`: EDA of the dataset, and training the baseline model.
* `1-Augment-Dataset.ipynb`: augmenting the dataset with rolling windows and saving trainning data in S3. 
* `2-Model Serving Client.ipynb`: local testing of baseline model using TF serving. 
* `3-SMILE eGeMAPSv01b Feature Extraction + Augmented Data.ipynb`: training the SMILE model with augmented data.
* `4-RollingWindow+LSTM+WN.ipynb`: training the dense + lstm model with rolling window and white noises. 
* `5-BERT_Transcription_text_training_MZ.ipynb`: training the BERT model with shuffled texts. 
* `6-Ensemble.ipynb`: blending the model predictions for ensemble and calibration graph. 