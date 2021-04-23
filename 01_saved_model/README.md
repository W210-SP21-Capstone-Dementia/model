This folder stores the models we trained. The four models we have in production are:
1. the baseline model `saved_model/base_line`
2. the LSTM model trained with white noise `saved_model/michael/rl_lstm_wn_0314`
3. the SMILE model we trained using eGeMAPS features `saved_model/michael/eGeMAPS`
4. the BERT model. It's too big for GitHub, download it from S3 `aws s3 sync s3://w210-audio-files-bucket/bert/ ~/model/saved_model/michael/BERT/ `.
