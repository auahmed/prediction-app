from flask import Flask, request
import os
from google.cloud import storage

app = Flask(__name__)

local_gcs = '/Users/x215839/python/prediction-app/gcs/'
model_file = 'data/test.json'


def get_model():
    """A function that will go and fetch the model from google cloud bucket
    """
    print('Getting model from GCS')
    bucket_name = 'beyond-analytics-247114-tf-test'
    # The "folder" where the files you want to download are

    # Create this folder locally
    if not os.path.exists(local_gcs):
        os.makedirs(local_gcs)
    else:
        print('gcs folder already exists')

    client = storage.Client()
    bucket = client.get_bucket(bucket_name)
    blob = bucket.get_blob(model_file)
    blob.download_to_filename(local_gcs+blob.name.rsplit('/', 1)[-1])
    print('Downloaded ' + model_file + ' from gcs successfully')


def get_prediction(input):
    """A function that will return ....

    Args:
        input: 
    """
    return 'prediction app running'

# [START index handler]
@app.route('/', methods=['GET'])
def index():
    if request.method == 'GET':
        return 'prediction app running'
# [END index handler]

# [START prediction handler]
@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        if os.path.exists(local_gcs+model_file.rsplit('/', 1)[-1]):
            print('Already have model file from gcs')
        else:
            get_model()

        predict = get_prediction(request.data)
        return predict, 200
# [END prediction handler]


app.run(host='0.0.0.0', port=8081)
