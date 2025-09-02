import sagemaker
from sagemaker.pytorch import PyTorch
import boto3
import os

# ----------------------------
# Setup
# ----------------------------
role = sagemaker.get_execution_role()
session = sagemaker.Session()

bucket = "cancer-prediction-data-arash"  
prefix = "data"                         
data_path = f"s3://{bucket}/{prefix}/breast_cancer.csv"

# ----------------------------
# Estimator (Training Job)
# ----------------------------
estimator = PyTorch(
    entry_point="train.py",          # our training script
    source_dir="src",                # folder where train.py is
    role=role,
    framework_version="1.12",        # PyTorch version
    py_version="py38",               # Python version
    instance_type="ml.m5.large",     # training instance
    instance_count=1,
    hyperparameters={
        "epochs": 20,
        "lr": 0.001,
    },
    disable_profiler=True
)

# ----------------------------
# Launch training
# ----------------------------
inputs = {"train": data_path}

estimator.fit(inputs)
