import pandas as pd
import argparse

def merge_verifier_with_regression():
    # Read the verifier CSV (with avg_yes_prob)
    verifier_df = pd.read_csv("verifier_scripts/deepswe-verifier-only-matching-pairs-qwen3-14b-lora64_N16_traindata_p2p-14B-75k-lr1en5-v1.csv")
    
    # Read the regression CSV
    regression_df = pd.read_csv("deepswe-swebv-eval-n16-verifier-v1-with-regression.csv")
    
    # Create merge keys
    verifier_df['merge_key'] = verifier_df['exp_names'] + '_' + verifier_df['docker_images']
    regression_df['merge_key'] = regression_df['exp_names'] + '_' + regression_df['docker_images']
    
    # Keep only necessary columns from regression df
    regression_subset = regression_df[['merge_key', 'regression_pass_count']].drop_duplicates()
    
    # Merge
    merged_df = verifier_df.merge(regression_subset, on='merge_key', how='left')
    
    # Fill NaN regression counts with 0
    merged_df['regression_pass_count'] = merged_df['regression_pass_count'].fillna(0)
    
    # Drop the merge key
    merged_df = merged_df.drop('merge_key', axis=1)
    
    # Save
    output_path = "verifier_with_regression_merged.csv"
    merged_df.to_csv(output_path, index=False)
    
    print(f"Merged CSV saved to: {output_path}")
    print(f"Shape: {merged_df.shape}")
    print(f"Rows with regression data: {(merged_df['regression_pass_count'] > 0).sum()}")
    print(f"Rows without regression data: {(merged_df['regression_pass_count'] == 0).sum()}")
    
    return merged_df

if __name__ == "__main__":
    df = merge_verifier_with_regression()
    
    # Show some statistics
    print("\nSample of merged data:")
    print(df[['docker_images', 'exp_names', 'avg_yes_prob', 'p2p_rates', 'regression_pass_count', 'gt_correct']].head())