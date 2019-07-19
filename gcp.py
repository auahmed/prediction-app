from google.cloud import bigquery, storage
import json
import os

dirpath = os.getcwd()
local_gcs = dirpath + '/gcs/'

def stream_bq(data):
    '''
    Args:
        data: json object to be streamed to BQ
    '''
    client = bigquery.Client()
    dataset_id = 'modelOutput'
    dataset_ref = client.dataset(dataset_id)
    table_ref = dataset_ref.table("modelOutputTable")

    # Format JSON object for insert to BQ
    newJson = json.dumps(data)
    jsonData = json.loads(newJson)

    table = client.get_table(table_ref)

    rows_to_insert = [jsonData]
    errors = client.insert_rows(table, rows_to_insert)
    if (errors != []):
        print(errors)
    assert errors == []

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
    blobs = bucket.list_blobs(prefix=location)
    for blob in blobs:
        blob.download_to_filename(
            local_gcs + saveTo + blob.name.rsplit('/', 1)[-1])