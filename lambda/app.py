import json
import boto3
import io
import PyPDF2
import re

runtime = boto3.client("sagemaker-runtime", region_name="us-east-2")
s3 = boto3.client("s3", region_name="us-east-2")

ENDPOINT_NAME = "skill-extract-hugging-face-endpoint"

# Words we don’t want to treat as skills
FILLER_WORDS = {
    "skills", "in", "and", "with", "at", "on", "for",
    ",", ".", ";", ":", "-", "the", "a", "an"
}

# Common false positives (degree names, random tokens, etc.)
FALSE_POSITIVES = {
    "be", "engineering", "engineer", "bachelor", "university",
    "education", "indian", "student"
}

def clean_token(token: str) -> str:
    token = token.strip()
    token = re.sub(r"[^a-zA-Z0-9+#]+", "", token)  # keep alphanumerics, +, #
    token_lower = token.lower()

    if not token or token_lower in FILLER_WORDS or token_lower in FALSE_POSITIVES:
        return ""
    return token


def extract_skills(ner_output):

    skills = []
    for item in ner_output:
        label = item.get("entity", "")
        token = item.get("word", "")

        if label == "I-Skills":
            cleaned = clean_token(token)
            if not cleaned:
                continue
            if token.startswith("##") and skills:
                skills[-1] += token[2:]  # merge subwords
            else:
                skills.append(cleaned)

    return skills


def chunk_text(text, max_chars=1000):

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

        s3_response = s3.get_object(Bucket=bucket, Key=key)
        pdf_content = s3_response["Body"].read()
        resume_text = ""

        # Extract text from PDF
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

        # Deduplicate + sort skills
        unique_skills = sorted(set(all_skills), key=str.lower)
        print("Extracted skills:", unique_skills)

        return {
            "statusCode": 200,
            "body": json.dumps({"skills": unique_skills}),
        }

    except Exception as e:
        print("Error:", str(e))
        return {
            "statusCode": 500,
            "body": json.dumps({"error": str(e)}),
        }
