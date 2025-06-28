import pandas as pd
import argparse

def merge_regression_data(verifier_csv, regression_csv, output_csv):
    """
    Merge verifier CSV with regression count data based on exp_name + docker_image.
    """
    # Read the CSVs
    verifier_df = pd.read_csv(verifier_csv)
    regression_df = pd.read_csv(regression_csv)
    
    # Create merge key for verifier data
    verifier_df['merge_key'] = verifier_df['exp_names'] + '_' + verifier_df['docker_images']
    
    # Assuming regression CSV has columns like 'exp_name', 'docker_image', 'regression_count'
    # Adjust column names as needed based on actual regression CSV structure
    if 'exp_name' in regression_df.columns and 'docker_image' in regression_df.columns:
        regression_df['merge_key'] = regression_df['exp_name'] + '_' + regression_df['docker_image']
    elif 'exp_names' in regression_df.columns and 'docker_images' in regression_df.columns:
        regression_df['merge_key'] = regression_df['exp_names'] + '_' + regression_df['docker_images']
    else:
        # Try to infer from available columns
        print("Available columns in regression CSV:", regression_df.columns.tolist())
        raise ValueError("Cannot find exp_name/docker_image columns in regression CSV")
    
    # Find the regression count column
    regression_count_col = None
    for col in regression_df.columns:
        if 'regression' in col.lower() and ('count' in col.lower() or 'test' in col.lower()):
            regression_count_col = col
            break
    
    if regression_count_col is None:
        print("Available columns in regression CSV:", regression_df.columns.tolist())
        raise ValueError("Cannot find regression count column in regression CSV")
    
    print(f"Using regression count column: {regression_count_col}")
    
    # Merge the dataframes
    merged_df = verifier_df.merge(
        regression_df[['merge_key', regression_count_col]], 
        on='merge_key', 
        how='left'
    )
    
    # Drop the merge_key column
    merged_df = merged_df.drop('merge_key', axis=1)
    
    # Rename regression count column if needed
    if regression_count_col != 'regression_count':
        merged_df = merged_df.rename(columns={regression_count_col: 'regression_count'})
    
    # Fill NaN values with 0 for missing regression counts
    merged_df['regression_count'] = merged_df['regression_count'].fillna(0)
    
    # Save the merged CSV
    merged_df.to_csv(output_csv, index=False)
    print(f"Merged CSV saved to: {output_csv}")
    
    # Print statistics
    print(f"\nMerge Statistics:")
    print(f"Original verifier rows: {len(verifier_df)}")
    print(f"Regression data rows: {len(regression_df)}")
    print(f"Merged rows: {len(merged_df)}")
    print(f"Rows with regression data: {(merged_df['regression_count'] > 0).sum()}")
    print(f"Rows without regression data: {(merged_df['regression_count'] == 0).sum()}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Merge verifier CSV with regression count data")
    parser.add_argument("verifier_csv", help="Path to verifier CSV file")
    parser.add_argument("regression_csv", help="Path to regression count CSV file")
    parser.add_argument("-o", "--output", default="merged_with_regression.csv", 
                        help="Output CSV file path (default: merged_with_regression.csv)")
    
    args = parser.parse_args()
    merge_regression_data(args.verifier_csv, args.regression_csv, args.output)