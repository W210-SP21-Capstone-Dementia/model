import os
import tempfile

import pytest

from flask_prediction import app

def test_file_upload():
    client = app.test_client() # you will need your flask app to create the test_client
    test_file = open('S001.wav', 'rb')
    data = {
        'file': test_file,
    }
    # note in that in the previous line you can use 'file' or whatever you want.
    # flask client checks for the tuple (<FileObject>, <String>)
    res = client.post('/predict', data=data) 
    assert res.status_code == 200

test_file_upload