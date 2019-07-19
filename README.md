# prediction-app

This application trains a Keras Tensorflow DNN model on US census data to predict income. Details on the dataset can be found at https://archive.ics.uci.edu/ml/datasets/Census+Income.

SHAP is used to generate explanations on predictions and these explanations are logged to BigQuery for analysis. Details on SHAP can be found here https://github.com/slundberg/shap.

The application is deployed using App Engine and predictions are generated on data sent via post requests.

## Run locally

To run locally, we will be required to have a json credential file.
This file can be obtained from: https://console.cloud.google.com/apis/credentials?project=beyond-analytics-247114&organizationId=0

```bash
virtualenv --python python3 env
source env/bin/activate
pip3 install -r requirements.txt
export GOOGLE_APPLICATION_CREDENTIALS=<json-file-with-credentials>
python3 main.py
deactivate
```
