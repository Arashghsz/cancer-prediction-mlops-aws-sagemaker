import sagemaker
from sagemaker.pytorch import PyTorch
import boto3
import os
import argparse

def train_model(mode='sagemaker', bucket=None):
    """
    Run a SageMaker training job, either locally or in the cloud
    """
    # Setup
    role = sagemaker.get_execution_role()
    session = sagemaker.Session()
    
    # Use provided bucket or default session bucket
    if not bucket:
        bucket = session.default_bucket()
    
    prefix = "data"
    data_path = f"s3://{bucket}/{prefix}/breast_cancer.csv"
    
    print(f"Mode: {mode}")
    print(f"Data path: {data_path}")
    
    # SageMaker Estimator
    estimator = PyTorch(
        entry_point="train.py",
        source_dir="src",
        role=role,
        framework_version="1.12",
        py_version="py38",
        instance_type="ml.m5.large" if mode == 'sagemaker' else "local",
        instance_count=1,
        hyperparameters={
            "epochs": 20,
            "lr": 0.001,
        },
        disable_profiler=True
    )
    
    # Launch training
    inputs = {"train": data_path}
    
    print(f"Starting {'local' if mode == 'local' else 'SageMaker'} training job...")
    estimator.fit(inputs)
    
    print(f"Training complete. Model saved to: {estimator.model_data}")
    return estimator

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type=str, choices=['local', 'sagemaker'], default='sagemaker',
                        help="Run training job locally or on SageMaker")
    parser.add_argument("--bucket", type=str, default=None,
                        help="S3 bucket name")
                        
    args = parser.parse_args()
    train_model(args.mode, args.bucket)
