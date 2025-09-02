import json
import boto3
import os

def lambda_handler(event, context):
    """
    Lambda function to invoke a SageMaker endpoint for cancer prediction
    """
    # Get the endpoint name from environment variable or use default
    endpoint_name = os.environ.get('ENDPOINT_NAME', 'cancer-prediction-model')
    
    try:
        # Parse the input
        if 'body' in event:
            body = json.loads(event['body'])
        else:
            body = event
            
        features = body.get('features', [])
        
        if not features:
            return {
                'statusCode': 400,
                'body': json.dumps({'error': 'No features provided'})
            }
            
        # Prepare the payload
        payload = {
            'features': features
        }
        
        # Create SageMaker runtime client
        runtime = boto3.client('sagemaker-runtime')
        
        # Invoke endpoint
        response = runtime.invoke_endpoint(
            EndpointName=endpoint_name,
            ContentType='application/json',
            Body=json.dumps(payload)
        )
        
        # Parse response
        result = json.loads(response['Body'].read().decode())
        
        # Return prediction
        return {
            'statusCode': 200,
            'headers': {
                'Content-Type': 'application/json',
                'Access-Control-Allow-Origin': '*'
            },
            'body': json.dumps(result)
        }
        
    except Exception as e:
        print(f"Error: {str(e)}")
        return {
            'statusCode': 500,
            'body': json.dumps({'error': str(e)})
        }
