import json
import boto3
import urllib.parse
import PyPDF2
import io

s3 = boto3.client('s3')

def handler(event, context):
    print("Received event: " + json.dumps(event, indent=2))

    # Extract bucket name and object key from the event
    bucket = event['Records'][0]['s3']['bucket']['name']
    key = event['Records'][0]['s3']['object']['key']
    key = urllib.parse.unquote_plus(key)

    try:
        # Get the PDF file from S3
        response = s3.get_object(Bucket=bucket, Key=key)
        pdf_content = response['Body'].read()

        # Read the PDF content
        pdf_reader = PyPDF2.PdfReader(io.BytesIO(pdf_content))
        text_content = ""
        for page in pdf_reader.pages:
            text_content += page.extract_text() + "\n"
        print("Extracted text: " + text_content)
        return {
            'statusCode': 200,
            'body': json.dumps({'text': text_content})
        }
    except Exception as e:
        print(e)
        print(f"Error processing object {key} from bucket {bucket}.")
        raise e

