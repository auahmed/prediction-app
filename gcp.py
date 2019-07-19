from google.cloud import bigquery
import json

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
