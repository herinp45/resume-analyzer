import json
import boto3
import urllib.parse
import PyPDF2
import io

s3 = boto3.client("s3")
runtime = boto3.client("sagemaker-runtime", region_name="us-east-2")

ENDPOINT_NAME = "skill-extract-hugging-face-endpoint"


def extract_skills_from_ner(ner_results):
    """Post-process Hugging Face NER output to extract clean skill strings."""
    skills = []
    current_skill = []

    for token in ner_results:
        word = token["word"]

        # Skip punctuation or filler words
        if word in [",", ".", ";", "in", "of"]:
            continue

        # Merge WordPiece tokens like ##WS -> AWS
        if word.startswith("##"):
            if current_skill:
                current_skill[-1] = current_skill[-1] + word[2:]
            continue

        # Start of a new skill
        if token["entity"].startswith("B-"):
            if current_skill:
                skills.append(" ".join(current_skill))
                current_skill = []
            current_skill.append(word)

        # Continuation of a skill
        elif token["entity"].startswith("I-"):
            current_skill.append(word)

    # Add last skill
    if current_skill:
        skills.append(" ".join(current_skill))

    return skills


def handler(event, context):
    print("Received event:", json.dumps(event, indent=2))

    try:
        # Extract bucket + object key from event
        bucket = event["Records"][0]["s3"]["bucket"]["name"]
        key = urllib.parse.unquote_plus(event["Records"][0]["s3"]["object"]["key"])

        # Fetch PDF file from S3
        response = s3.get_object(Bucket=bucket, Key=key)
        pdf_content = response["Body"].read()

        # Extract text from PDF
        pdf_reader = PyPDF2.PdfReader(io.BytesIO(pdf_content))
        text_content = ""
        for page in pdf_reader.pages:
            text_content += page.extract_text() + "\n"

        print("Extracted text (first 500 chars):", text_content[:500])

        # Invoke SageMaker endpoint
        sm_response = runtime.invoke_endpoint(
            EndpointName=ENDPOINT_NAME,
            ContentType="application/json",
            Body=json.dumps({"inputs": text_content}),
        )

        ner_output = json.loads(sm_response["Body"].read().decode())
        print("Raw model output:", ner_output[:10])  # print first 10 tokens

        # Post-process to clean skills
        skills = extract_skills_from_ner(ner_output)
        print("Final skills:", skills)

        return {
            "statusCode": 200,
            "body": json.dumps({"skills": skills}),
        }

    except Exception as e:
        print("Error:", str(e))
        raise e
