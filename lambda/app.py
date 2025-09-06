import json
import boto3
import io
import PyPDF2

runtime = boto3.client("sagemaker-runtime", region_name="us-east-2")
s3 = boto3.client("s3", region_name="us-east-2")
ENDPOINT_NAME = "skill-extract-hugging-face-endpoint"

FILLER_WORDS = {"skills", "in", "and", "with", "at", "on", "for", ",", ".", ";", ":"}

def extract_skills(ner_output):
    """
    Extract skills from the NER model output.
    """
    skills = []
    for item in ner_output:
        label = item.get("entity", "")
        token = item.get("word", "")

        if label == "I-Skills":
            if token.lower() in FILLER_WORDS:
                continue
            if token.startswith("##") and skills:
                skills[-1] += token[2:]  # merge subwords
            else:
                skills.append(token)
    return skills


def handler(event, context):
    print("Received event:", json.dumps(event, indent=2))

    try:
        bucket = event["Records"][0]["s3"]["bucket"]["name"]
        key = event["Records"][0]["s3"]["object"]["key"]

        s3_response = s3.get_object(Bucket=bucket, Key=key)
        pdf_content = s3_response["Body"].read()
        resume_text = ""
        pdf_reader = PyPDF2.PdfReader(io.BytesIO(pdf_content))
        for page in pdf_reader.pages:
            resume_text += page.extract_text() + "\n"

        # Invoke SageMaker endpoint
        sm_response = runtime.invoke_endpoint(
            EndpointName=ENDPOINT_NAME,
            ContentType="application/json",
            Body=json.dumps({"inputs": resume_text}),
        )

        ner_output = json.loads(sm_response["Body"].read().decode())
        print("Raw model output sample:", ner_output[:5])  # debug

        # Use helper function
        skills = extract_skills(ner_output)
        print("Extracted skills:", skills)

        return {
            "statusCode": 200,
            "body": json.dumps({"skills": skills}),
        }

    except Exception as e:
        print("Error:", str(e))
        return {
            "statusCode": 500,
            "body": json.dumps({"error": str(e)}),
        }
