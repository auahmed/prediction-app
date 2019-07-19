# prediction-app

# Run locally

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