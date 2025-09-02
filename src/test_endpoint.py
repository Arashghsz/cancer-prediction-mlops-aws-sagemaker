import argparse
import boto3
import json
import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer

def test_endpoint(endpoint_name, sample_count=5):
    """
    Test a SageMaker endpoint with sample data
    """
    # Load sample data (can be replaced with your own test data)
    data = load_breast_cancer()
    features = data.data[:sample_count]
    actual_labels = data.target[:sample_count]
    
    # Create SageMaker runtime client
    runtime = boto3.client('sagemaker-runtime')
    
    print(f"Testing endpoint {endpoint_name} with {sample_count} samples")
    print("-" * 50)
    
    for i, (feature_vector, actual_label) in enumerate(zip(features, actual_labels)):
        # Prepare the request payload
        payload = {
            "features": feature_vector.tolist()
        }
        
        # Invoke endpoint
        response = runtime.invoke_endpoint(
            EndpointName=endpoint_name,
            ContentType='application/json',
            Body=json.dumps(payload)
        )
        
        # Parse response
        result = json.loads(response['Body'].read().decode())
        predicted_label = result['prediction'][0][0]
        
        # Print results
        print(f"Sample {i+1}:")
        print(f"  Predicted: {predicted_label} (Class: {int(round(predicted_label))})")
        print(f"  Actual:    {actual_label}")
        print(f"  Match:     {'✓' if int(round(predicted_label)) == actual_label else '✗'}")
        print("-" * 30)
        
    print("Test completed!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--endpoint-name", type=str, default="cancer-prediction-model",
                        help="Name of the SageMaker endpoint to test")
    parser.add_argument("--samples", type=int, default=5,
                        help="Number of samples to test")
    
    args = parser.parse_args()
    test_endpoint(args.endpoint_name, args.samples)
