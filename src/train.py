import argparse
import os
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
import boto3

# -----------------------------
# Simple Neural Net
# -----------------------------
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

# -----------------------------
# Training function
# -----------------------------
def train_model(data_path, model_dir, epochs=20, lr=0.001):
    # Load dataset
    print(f"Loading dataset from {data_path}...")
    df = pd.read_csv(data_path)

    X = df.drop("target", axis=1).values
    y = df["target"].values

    # Scale features
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Convert to torch tensors
    X_train = torch.tensor(X_train, dtype=torch.float32)
    y_train = torch.tensor(y_train, dtype=torch.float32).view(-1, 1)
    X_test = torch.tensor(X_test, dtype=torch.float32)
    y_test = torch.tensor(y_test, dtype=torch.float32).view(-1, 1)

    # Model, loss, optimizer
    model = CancerNet(input_dim=X.shape[1])
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    # Training loop
    for epoch in range(epochs):
        optimizer.zero_grad()
        outputs = model(X_train)
        loss = criterion(outputs, y_train)
        loss.backward()
        optimizer.step()

        if (epoch+1) % 5 == 0:
            print(f"Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}")

    with torch.no_grad():
        preds = model(X_test)
        preds = (preds > 0.5).float()

        # Convert both to numpy arrays for sklearn
        y_true = y_test.numpy()
        y_pred = preds.numpy()

        acc = accuracy_score(y_true, y_pred)
        print(f"Test Accuracy: {acc:.4f}")


    # Save model
    os.makedirs(model_dir, exist_ok=True)
    model_path = os.path.join(model_dir, "model.pth")
    torch.save(model.state_dict(), model_path)
    print(f"Model saved at {model_path}")

# -----------------------------
# Main (SageMaker entry point)
# -----------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # SageMaker passes model directory here
    parser.add_argument("--model-dir", type=str, default=os.environ.get("SM_MODEL_DIR", "/opt/ml/model"))
    parser.add_argument("--data-path", type=str, default=os.environ.get("SM_CHANNEL_TRAIN", "../data/breast_cancer.csv"))
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--lr", type=float, default=0.001)

    args = parser.parse_args()

    train_model(args.data_path, args.model_dir, args.epochs, args.lr)
