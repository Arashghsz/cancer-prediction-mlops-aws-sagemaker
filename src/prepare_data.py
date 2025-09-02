from sklearn.datasets import load_breast_cancer
import pandas as pd

# Load dataset
data = load_breast_cancer()
df = pd.DataFrame(data.data, columns=data.feature_names)
df['target'] = data.target

# Save to CSV
df.to_csv("data/breast_cancer.csv", index=False)
print("Dataset saved as breast_cancer.csv")
