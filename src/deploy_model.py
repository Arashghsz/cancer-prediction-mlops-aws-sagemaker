import argparse
import boto3
import sagemaker
from sagemaker.pytorch import PyTorchModel

def deploy_model(model_name, model_s3_path, role, instance_type="ml.t2.medium"):
    """
    Deploy a PyTorch model to a SageMaker endpoint
    """
    print(f"Deploying model {model_name} from {model_s3_path}")
    
    # Initialize SageMaker session
    session = sagemaker.Session()
    
    # Create PyTorch model
    pytorch_model = PyTorchModel(
        model_data=model_s3_path,
        role=role,
        entry_point="predict.py",  # Script for inference
        source_dir="src",          # Directory containing the script
        framework_version="1.12",
        py_version="py38",
    )
    
    # Deploy to endpoint
    predictor = pytorch_model.deploy(
        initial_instance_count=1,
        instance_type=instance_type,
        endpoint_name=model_name
    )
    
    print(f"Model deployed to endpoint: {model_name}")
    return predictor

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-name", type=str, required=True, 
                        help="Name for the SageMaker endpoint")
    parser.add_argument("--model-s3-path", type=str, 
                        default=None, 
                        help="S3 path to model artifact. If not provided, will use the latest training job.")
    parser.add_argument("--instance-type", type=str, 
                        default="ml.t2.medium", 
                        help="Instance type for the endpoint")
    
    args = parser.parse_args()
    
    # If model path not provided, get it from the latest training job
    if not args.model_s3_path:
        sm_client = boto3.client('sagemaker')
        training_job_name = sm_client.list_training_jobs(
            SortBy='CreationTime',
            SortOrder='Descending',
            MaxResults=1
        )['TrainingJobSummaries'][0]['TrainingJobName']
        
        response = sm_client.describe_training_job(TrainingJobName=training_job_name)
        args.model_s3_path = response['ModelArtifacts']['S3ModelArtifacts']
    
    # Get role ARN
    role = sagemaker.get_execution_role()
    
    deploy_model(args.model_name, args.model_s3_path, role, args.instance_type)
