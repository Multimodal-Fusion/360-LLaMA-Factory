import pandas as pd
import glob
import os

# Set pandas options to display the full table without truncation.
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
pd.set_option('display.max_colwidth', None)

N = 500

def accuracy_topkyesprob_then_p2p(df, k=3):
    df = df.copy()  
    # df['p2p_rate_len'] = df['p2p_rate'].apply(len)
    df['p2p_rate_len'] = df['p2p_rates']
    
    grouped = df.groupby("docker_images")
    final_corrects = []
    
    for docker_image, group in grouped:
        topk = group.sort_values(by="avg_yes_prob", ascending=False).head(k)
        chosen_row = topk.loc[topk["p2p_rate_len"].idxmax()]
        final_corrects.append(chosen_row["gt_correct"])
    
    if not final_corrects:
        return 0.0
    return sum(final_corrects) / N

def get_topk_with_ties(group, key, k, ascending=False):
    group_sorted = group.sort_values(by=key, ascending=ascending)
    if len(group_sorted) <= k:
        return group_sorted
    kth_value = group_sorted.iloc[k-1][key]
    return group_sorted[group_sorted[key] >= kth_value]

def accuracy_topkp2p_then_yesprob(df, k=3):
    df = df.copy() 
    # df['p2p_rate_len'] = df['p2p_rate'].apply(len)
    df['p2p_rate_len'] = df['p2p_rates']
    
    grouped = df.groupby("docker_images")
    final_corrects = []
    
    for docker_image, group in grouped:
        topk = get_topk_with_ties(group, "p2p_rate_len", k)
        chosen_row = topk.loc[topk["avg_yes_prob"].idxmax()]
        final_corrects.append(chosen_row["gt_correct"])
    
    if not final_corrects:
        return 0.0
    return sum(final_corrects) / N

def accuracy_topkp2p_then_yesprob_alt(df, k=3):
    df = df.copy()
    # df['p2p_rate_len'] = df['p2p_rate'].apply(len)
    df['p2p_rate_len'] = df['p2p_rates']
    
    def get_topk_with_ties_inner(group, col, k):
        group_sorted = group.sort_values(by=col, ascending=False)
        if len(group_sorted) <= k:
            return group_sorted
        kth_value = group_sorted.iloc[k-1][col]
        return group_sorted[group_sorted[col] >= kth_value]
    
    grouped = df.groupby("docker_images")
    final_corrects = []
    
    for docker_image, group in grouped:
        steps_min = group['num_steps'].min()
        steps_max = group['num_steps'].max()
        if steps_max == steps_min:
            group['num_steps_norm'] = 0.0
        else:
            group['num_steps_norm'] = (group['num_steps'] - steps_min) / (steps_max - steps_min)
        
        group['adjusted_score'] = group['avg_yes_prob']  # adjust if needed
        
        topk = get_topk_with_ties_inner(group, "p2p_rate_len", k)
        chosen_row = topk.loc[topk["adjusted_score"].idxmax()]
        final_corrects.append(chosen_row["gt_correct"])
    
    if not final_corrects:
        return 0.0
    return sum(final_corrects) / len(final_corrects)

def run_aggregation2(df):
    # Initialize a dictionary to store metrics.
    metrics = {}
    
    # Group by 'docker_images' and select the row with maximum avg_yes_prob per group.
    aggregated = df.groupby("docker_images", as_index=False).apply(
        lambda group: group.loc[group["avg_yes_prob"].idxmax()]
    ).reset_index(drop=True)
    overall_accuracy = aggregated["gt_correct"].mean()
    total_trajs = len(df)
    
    # Compute RM accuracy based on a simple threshold (0.5) for avg_yes_prob.
    df['predicted'] = (df['avg_yes_prob'] > 0.5).astype(int)
    df['gt_correct'] = df['gt_correct'].astype(int)
    df['correct_prediction'] = (df['predicted'] == df['gt_correct']).astype(int)
    overall_rm_accuracy = df['correct_prediction'].mean()
    
    # Compute the top-k accuracies.
    k_yesprob = 10
    acc_topk_yesprob = accuracy_topkyesprob_then_p2p(df, k=k_yesprob)
    
    k_p2p = 1
    acc_topk_p2p = accuracy_topkp2p_then_yesprob(df, k=k_p2p)
    
    metrics["total_trajs"] = total_trajs
    metrics["overall_accuracy"] = overall_accuracy
    metrics["overall_rm_accuracy"] = overall_rm_accuracy
    metrics["accuracy_topk_yesprob"] = acc_topk_yesprob
    metrics["accuracy_topk_p2p"] = acc_topk_p2p
    
    return metrics

def run_aggregation(df):
    # Filter to the first 100 unique docker images.
    first_100_dockers = df['docker_images'].drop_duplicates().iloc[:100]
    df = df[df['docker_images'].isin(first_100_dockers)]
    
    # Initialize a dictionary to store metrics.
    metrics = {}
    
    # Group by 'docker_images' and select the row with maximum avg_yes_prob per group.
    aggregated = df.groupby("docker_images", as_index=False).apply(
        lambda group: group.loc[group["avg_yes_prob"].idxmax()]
    ).reset_index(drop=True)
    
    overall_accuracy = aggregated["gt_correct"].mean()
    total_trajs = len(df)
    
    # Compute RM accuracy based on a simple threshold (0.5) for avg_yes_prob.
    df['predicted'] = (df['avg_yes_prob'] > 0.5).astype(int)
    df['gt_correct'] = df['gt_correct'].astype(int)
    df['correct_prediction'] = (df['predicted'] == df['gt_correct']).astype(int)
    overall_rm_accuracy = df['correct_prediction'].mean()
    
    # Compute the top-k accuracies.
    k_yesprob = 10
    acc_topk_yesprob = accuracy_topkyesprob_then_p2p(df, k=k_yesprob)
    
    k_p2p = 1
    acc_topk_p2p = accuracy_topkp2p_then_yesprob(df, k=k_p2p)
    
    metrics["total_trajs"] = total_trajs
    metrics["overall_accuracy"] = overall_accuracy
    metrics["overall_rm_accuracy"] = overall_rm_accuracy
    metrics["accuracy_topk_yesprob"] = acc_topk_yesprob
    metrics["accuracy_topk_p2p"] = acc_topk_p2p
    
    return metrics

def main():
    # Collect results in a list of dictionaries.
    results = []
    
    # Get all CSV files starting with "results" in the given directory.
    csv_files = glob.glob("/home/ubuntu/360-LLaMA-Factory/*.csv")
    # Sort the files by modification time (newest first) and take the most recent 10.
    csv_files = sorted(csv_files, key=lambda x: os.path.getmtime(x), reverse=True)
    most_recent_files = csv_files[:40]
    
    # df2 = pd.read_csv("./src/r2e_edits/agenthub/train/results_traj_verifiernew_temp09_fixes_p2p-14B-32k-lora64-1en5-v1.csv")
    # df3 = pd.read_csv("./src/r2e_edits/agenthub/train/results_traj_verifiernew_p2p-14B_32k-t0-v1.csv")

    for file in most_recent_files:
        print(f"\nProcessing file: {file}")
        # Initialize default metrics.
        result_entry = {
            "total_trajs": None,
            "overall_accuracy": None,
            "overall_rm_accuracy": None,
            "accuracy_topk_yesprob": None,
            "accuracy_topk_p2p": None,
            "file": os.path.basename(file),
        }
        try:
            df = pd.read_csv(file)
            # df = pd.concat([df, df2, df3], ignore_index=True)
            metrics = run_aggregation(df)
            result_entry.update(metrics)
        except Exception as e:
            print(f"Error processing file {file}: {e}")
        results.append(result_entry)
    
    # Create a DataFrame to display the results as a table.
    results_df = pd.DataFrame(results)
    print("\nFinal Results Table:")
    print(results_df.to_string())

if __name__ == '__main__':
    main()