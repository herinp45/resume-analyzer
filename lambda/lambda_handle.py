import os
import json
import boto3

dynamodb = boto3.resource("dynamodb", region_name="us-east-2")
table = dynamodb.Table("ResumeAnalysis")

def lambda_handler(event, context):
    # For API Gateway proxy integration
    resume_id = event.get("pathParameters", {}).get("resumeId")

    if not resume_id:
        return {
            "statusCode": 400,
            "headers": {"Access-Control-Allow-Origin": "*"},
            "body": json.dumps({"error": "Missing resumeId in path"})
        }

    try:
        response = table.get_item(Key={"resumeId": resume_id})
        item = response.get("Item")

        if not item:
            return {
                "statusCode": 404,
                "headers": {"Access-Control-Allow-Origin": "*"},
                "body": json.dumps({"error": "Resume not found"})
            }

        return {
            "statusCode": 200,
            "headers": {"Access-Control-Allow-Origin": "*"},
            "body": json.dumps({"resumeId": resume_id, "skills": item.get("skills", [])})
        }

    except Exception as e:
        return {
            "statusCode": 500,
            "headers": {"Access-Control-Allow-Origin": "*"},
            "body": json.dumps({"error": str(e)})
        }
