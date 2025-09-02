import os
import torch
import torch.nn as nn
import json
import numpy as np

# Same model class as in train.py
class CancerNet(nn.Module):
    def __init__(self, input_dim):
        super(CancerNet, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.layers(x)

# Number of features in the breast cancer dataset
INPUT_DIM = 30  # Wisconsin breast cancer dataset has 30 features

def model_fn(model_dir):
    """
    Load the PyTorch model from the model_dir
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CancerNet(input_dim=INPUT_DIM)
    
    with open(os.path.join(model_dir, "model.pth"), "rb") as f:
        model.load_state_dict(torch.load(f, map_location=device))
    
    model.to(device)
    model.eval()
    return model

def input_fn(request_body, request_content_type):
    """
    Deserialize and prepare the prediction input
    """
    if request_content_type == "application/json":
        input_data = json.loads(request_body)
        features = input_data.get("features", input_data)  # Handle both {"features": [...]} and direct array
        
        # Convert to PyTorch tensor
        features_tensor = torch.tensor(features, dtype=torch.float32)
        if len(features_tensor.shape) == 1:
            features_tensor = features_tensor.unsqueeze(0)  # Add batch dimension if not present
            
        return features_tensor
    else:
        raise ValueError(f"Unsupported content type: {request_content_type}")

def predict_fn(input_data, model):
    """
    Apply model to the input data
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    input_data = input_data.to(device)
    
    with torch.no_grad():
        output = model(input_data)
    
    # Convert to binary prediction (0 or 1)
    prediction = (output > 0.5).float()
    return prediction

def output_fn(prediction, response_content_type):
    """
    Serialize and prepare the prediction output
    """
    if response_content_type == "application/json":
        response = prediction.cpu().numpy().tolist()
        return json.dumps({"prediction": response})
    else:
        raise ValueError(f"Unsupported content type: {response_content_type}")
