import os
import json
import boto3

runtime = boto3.client("sagemaker-runtime")
ENDPOINT_NAME = os.environ.get("ENDPOINT_NAME")  # e.g. "cancer-prediction-skl-1756892108"

def lambda_handler(event, context):
    try:
        # parse body (API Gateway proxy or direct invocation)
        if isinstance(event, dict) and "body" in event and event["body"]:
            body = json.loads(event["body"])
        else:
            body = event if isinstance(event, dict) else json.loads(event)

        # accept either {"features":[...]} or {"instances":[[...], ...]} or raw list
        instances = None
        if isinstance(body, dict):
            instances = body.get("instances") or body.get("features") or body.get("data")
        else:
            instances = body

        if instances is None:
            return {"statusCode": 400, "body": json.dumps({"error": "No features provided"})}

        # normalize to list of instances
        if isinstance(instances, list) and len(instances) > 0 and not isinstance(instances[0], list):
            payload_instances = [instances]
        else:
            payload_instances = instances

        payload = {"instances": payload_instances}

        resp = runtime.invoke_endpoint(
            EndpointName=ENDPOINT_NAME,
            ContentType="application/json",
            Body=json.dumps(payload)
        )

        resp_body = resp["Body"].read().decode()
        parsed = json.loads(resp_body)

        return {
            "statusCode": 200,
            "headers": {
                "Content-Type": "application/json",
                "Access-Control-Allow-Origin": "*"
            },
            "body": json.dumps({"endpoint_response": parsed})
        }

    except Exception as e:
        return {"statusCode": 500, "body": json.dumps({"error": str(e)})}