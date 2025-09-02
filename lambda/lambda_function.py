import json
import boto3

runtime = boto3.client('sagemaker-runtime')
ENDPOINT_NAME = "cancer-prediction-endpoint"

def lambda_handler(event, context):
    features = event["features"]
    payload = ",".join([str(x) for x in features])

    response = runtime.invoke_endpoint(
        EndpointName=ENDPOINT_NAME,
        ContentType="text/csv",
        Body=payload
    )

    result = response['Body'].read().decode('utf-8')
    return {
        "statusCode": 200,
        "body": json.dumps({"prediction": float(result)})
    }
