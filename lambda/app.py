import json
from idlelib.rpc import response_queue

import boto3
import urllib.parse

s3 = boto3.client('s3')
texract = boto3.client('textract')

def handler(event, context):
    print("Received event: " + json.dumps(event, indent=2))

    # Extract bucket name and object key from the event
    bucket = event['Records'][0]['s3']['bucket']['name']
    key = event['Records'][0]['s3']['object']['key']

    # URL decode
    response = texract.detect_document_text(
        Document={
            'S3Object': {
                'Bucket': bucket,
                'Name': key
            }
        }
    )
    print("Textract response: " + json.dumps(response, indent=2))

    # Extract detected text
    detected_text = "".join(item['Text'] for item in response['Blocks'] if item['BlockType'] == 'LINE')

    return {
        'statusCode': 200,
        'body': json.dumps({'detected_text': detected_text})
    }