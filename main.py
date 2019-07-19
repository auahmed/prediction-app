from flask import Flask, request
import os
from google.cloud import storage
from gcp import stream_bq
from model import _train, _getPrediction

app = Flask(__name__)

dirpath = os.getcwd()
local_gcs = dirpath + '/gcs/'
model_file = 'data/test.json'
training_file = 'data'

def get_gcs(location, saveTo):
  """A function to get files from gcs
  """
  print('getting file(s) for ' + location)
  bucket_name = 'beyond-analytics-247114-tf-test'

  # Create this folder locally
  if not os.path.exists(local_gcs + saveTo):
    os.makedirs(local_gcs + saveTo)
  else:
    print('folder already exists: ' + local_gcs + saveTo)

  client = storage.Client()
  bucket = client.get_bucket(bucket_name)
  # List blobs iterate in folder 
  blobs=bucket.list_blobs(prefix=location)
  for blob in blobs:
    blob.download_to_filename(local_gcs + saveTo + blob.name.rsplit('/', 1)[-1])


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
        # Obtain training data
        if os.path.exists(local_gcs+'data'):
            print('Already have training data')
        else:
            get_gcs('data', 'data/')

        # Train Data
        if os.path.exists(local_gcs + 'model.json'):
            print('Already trained the data')
        else:
            model_content = _train()
            f = open(local_gcs + 'model.json', "w+")
            f.write(model_content)
            f.close()

        predict = get_prediction(request.data)

        # Send prediction data to Big Query
        # stream_bq(predict)
        return predict, 200
# [END prediction handler]
