from datasets import load_dataset
import pandas as pd

# Download the dataset from HuggingFace
dataset = load_dataset("r2e-edits/deepswe-swebv-eval-n16-verifier-v1-with-regression")

# Convert to pandas DataFrame
df = dataset['train'].to_pandas()

# Save to CSV
output_path = "deepswe-swebv-eval-n16-verifier-v1-with-regression.csv"
df.to_csv(output_path, index=False)

print(f"Dataset saved to: {output_path}")
print(f"Shape: {df.shape}")
print(f"Columns: {df.columns.tolist()}")