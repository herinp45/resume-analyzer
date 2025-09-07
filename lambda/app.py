import json
import boto3
import io
import PyPDF2
import re
import os

runtime = boto3.client("sagemaker-runtime", region_name="us-east-2")
s3 = boto3.client("s3", region_name="us-east-2")
dynamodb = boto3.resource("dynamodb", region_name="us-east-2")
table = dynamodb.Table("ResumeAnalysis")

ENDPOINT_NAME = "skill-extract-hugging-face-endpoint"

# Stopwords / false positives that should not appear as skills
STOPWORDS = {
    "be", "engineering", "engineer", "bachelor", "university", "education",
    "indian", "student", "students", "teaching", "mentoring", "course",
    "weekly", "assignments", "academic", "grad", "date", "time", "help", "of",
    "to", "in", "on", "with", "at", "by", "like", "approaches", "materialills"
}

def clean_token(token: str) -> str:
    """
    Normalize and filter tokens that are likely false skills.
    """
    token = token.strip()
    token = re.sub(r"[^a-zA-Z0-9+#]+", "", token)  # remove symbols except +, #
    token_lower = token.lower()

    # Drop empty or stopwords
    if not token or token_lower in STOPWORDS:
        return ""

    # Drop numbers unless it's like C# or C++
    if token.isdigit():
        return ""

    # Drop short junk (len=1) unless it's C, R, or Go
    if len(token) == 1 and token not in {"C", "R", "Go"}:
        return ""

    # Normalize acronyms (e.g., SQL, AWS)
    if token.isupper():
        return token

    # Capitalize for consistency
    return token.capitalize()


def extract_skills(ner_output):
    """
    Extract and clean skills from model output.
    """
    skills = []
    for item in ner_output:
        label = item.get("entity", "")
        token = item.get("word", "")

        if label == "I-Skills":
            cleaned = clean_token(token)
            if not cleaned:
                continue
            if token.startswith("##") and skills:
                # merge subwords like "##Script" → "JavaScript"
                skills[-1] += token[2:]
            else:
                skills.append(cleaned)

    return skills


def chunk_text(text, max_chars=1000):
    """
    Break text into chunks so the model doesn't exceed token limits.
    """
    words = text.split()
    chunks, current = [], []

    for word in words:
        if sum(len(w) for w in current) + len(word) + len(current) > max_chars:
            chunks.append(" ".join(current))
            current = [word]
        else:
            current.append(word)

    if current:
        chunks.append(" ".join(current))

    return chunks


def handler(event, context):
    print("Received event:", json.dumps(event, indent=2))

    try:
        bucket = event["Records"][0]["s3"]["bucket"]["name"]
        key = event["Records"][0]["s3"]["object"]["key"]

        resumeId = os.path.splitext(key.split("/")[-1])[0]

    # Get PDF from S3
        s3_response = s3.get_object(Bucket=bucket, Key=key)
        pdf_content = s3_response["Body"].read()

        resume_text = ""
        pdf_reader = PyPDF2.PdfReader(io.BytesIO(pdf_content))
        for page in pdf_reader.pages:
            text = page.extract_text()
            if text:
                resume_text += text + "\n"

        # Process in chunks
        resume_chunks = chunk_text(resume_text)
        all_skills = []

        for chunk in resume_chunks:
            sm_response = runtime.invoke_endpoint(
                EndpointName=ENDPOINT_NAME,
                ContentType="application/json",
                Body=json.dumps({"inputs": chunk}),
            )
            ner_output = json.loads(sm_response["Body"].read().decode())
            all_skills.extend(extract_skills(ner_output))

        # Deduplicate + sort
        unique_skills = sorted(set(all_skills), key=str.lower)
        print("Extracted skills:", unique_skills)

        # Store in DynamoDB
        table.put_item(
            Item={
                "resumeId": resumeId,
                "skills": unique_skills,
            }
        )

        return {
            "statusCode": 200,
            "headers": {
                "Content-Type": "application/json",
                "Access-Control-Allow-Origin": "*",
            },
            "body": json.dumps({"skills": unique_skills}),
        }

    except Exception as e:
        print("Error:", str(e))
        return {
            "statusCode": 500,
            "headers": {
                "Content-Type": "application/json",
                "Access-Control-Allow-Origin": "*",
            },
            "body": json.dumps({"error": str(e)}),
        }
