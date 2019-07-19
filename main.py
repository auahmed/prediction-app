from flask import Flask, request
import os
import json
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
            print('Writing model to file ' + local_gcs + 'model.json')
            f.write(str(model_content))
            f.close()

        try:
          model_content
        except NameError:
          try:
            with open(local_gcs + 'model.json') as jsonfile:
                model_content = json.load(jsonfile)
            print('Loaded model JSON')
          except Exception as e:
            print('Unable to load model json from local')
            print(e)
            return 'Unable to load model json from local', 500

        predict = _getPrediction(model_content[0], request.get_json(), model_content[1])

        print('Successfully printed for input: ' + str(predict))

        # Send prediction data to Big Query
        stream_bq(predict)
        print('Successfully loaded prediction to Big Query')

        return predict, 200
# [END prediction handler]

app.run(host='0.0.0.0', port=8080)