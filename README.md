# cancer-prediction-mlops-aws-sagemaker
# Cancer Prediction MLOps on AWS

This repository implements a full MLOps workflow:

- Data stored in S3
- PyTorch model trained in SageMaker
- Model deployed as SageMaker real-time endpoint
- Endpoint accessed via Lambda + API Gateway
- CI/CD with GitHub + CodePipeline + CodeBuild

## Folder Structure
- src/ : training scripts
- lambda/ : Lambda function
- notebooks/ : optional exploration
- buildspec.yml : CI/CD
- requirements.txt : dependencies

## Usage
1. Push code to GitHub
2. Trigger CI/CD pipeline
3. Training job runs automatically
4. Endpoint updated
5. Call endpoint via API Gateway

