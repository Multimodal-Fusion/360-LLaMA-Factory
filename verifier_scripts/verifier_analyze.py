import pandas as pd
import glob

N = 500
def accuracy_topkyesprob_then_p2p(df, k=3):
    df = df.copy()  
    # df['p2p_rate_len'] = df['p2p_rate'].apply(len)
    df['p2p_rate_len'] = df['p2p_rates']
    
    grouped = df.groupby("docker_images")
    final_corrects = []

    for docker_image, group in grouped:
        # Sort by avg_yes_prob descending and take top k rows
        topk = group.sort_values(by="avg_yes_prob", ascending=False).head(k)

        # Among the top k, select the row with the highest p2p_rate_len
        chosen_row = topk.loc[topk["p2p_rate_len"].idxmax()]
        final_corrects.append(chosen_row["gt_correct"])

    # Overall accuracy
    if not final_corrects:
        return 0.0
    return sum(final_corrects) / N
    
def accuracy_topkp2p_then_yesprob(df, k=3):
    df = df.copy() 
    # df['p2p_rate_len'] = df['p2p_rate'].apply(len)
    df['p2p_rate_len'] = df['p2p_rates']    
    
    grouped = df.groupby("docker_images")
    final_corrects = []

    for docker_image, group in grouped:
        # Sort by p2p_rate_len descending and take the top k rows
        # topk = group.sort_values(by="p2p_rate_len", ascending=False).head(k)
        topk = get_topk_with_ties(group, "p2p_rate_len", k)
        
        # Among the top k, select the row with the highest avg_yes_prob
        chosen_row = topk.loc[topk["avg_yes_prob"].idxmax()]
        
        final_corrects.append(chosen_row["gt_correct"])

    # Overall accuracy
    if not final_corrects:
        return 0.0
    return sum(final_corrects) / N

def accuracy_topkp2p_then_yesprob_alt(df, k=3):
    df = df.copy()
    # df['p2p_rate_len'] = df['p2p_rate'].apply(len)
    df['p2p_rate_len'] = df['p2p_rates']
    
    # Helper function to get top-k rows with ties based on a given column.
    def get_topk_with_ties(group, col, k):
        group_sorted = group.sort_values(by=col, ascending=False)
        if len(group_sorted) <= k:
            return group_sorted
        kth_value = group_sorted.iloc[k-1][col]
        return group_sorted[group_sorted[col] >= kth_value]
    
    grouped = df.groupby("docker_images")
    final_corrects = []
    
    for docker_image, group in grouped:
        # Normalize num_steps for the group.
        steps_min = group['num_steps'].min()
        steps_max = group['num_steps'].max()
        if steps_max == steps_min:
            group['num_steps_norm'] = 0.0
        else:
            group['num_steps_norm'] = (group['num_steps'] - steps_min) / (steps_max - steps_min)
        
        # Compute the adjusted score.
        group['adjusted_score'] = group['avg_yes_prob'] #- group['num_steps_norm']
        
        # Select top k rows based on p2p_rate_len (keeping ties).
        topk = get_topk_with_ties(group, "p2p_rate_len", k)
        
        # Among these top rows, select the row with the highest adjusted_score.
        chosen_row = topk.loc[topk["adjusted_score"].idxmax()]
        final_corrects.append(chosen_row["gt_correct"])
    
    if not final_corrects:
        return 0.0
    return sum(final_corrects) / len(final_corrects)
    
def get_topk_with_ties(group, key, k, ascending=False):
    # Sort the group by 'p2p_rate_len' in descending order.
    group_sorted = group.sort_values(by=key, ascending=ascending)
    if len(group_sorted) <= k:
        return group_sorted
    kth_value = group_sorted.iloc[k-1][key]
    # Keep all rows where p2p_rate_len is >= kth_value
    return group_sorted[group_sorted[key] >= kth_value]

# # Load the CSV file into a DataFrame
# # df = pd.read_csv("./src/r2e_edits/agenthub/train/results_traj_verifier-14B-v2.csv")
# # df = pd.read_csv("./src/r2e_edits/agenthub/train/results_traj_verifiernew-14B-v3.csv")
# df = pd.read_csv("./src/r2e_edits/agenthub/train/results_traj_verifiernew-32B_32k-v2.csv")

# # Group by 'docker_images' and select the row with the maximum avg_yes_prob from each group
# aggregated = df.groupby("docker_images", as_index=False).apply(
#     lambda group: group.loc[group["avg_yes_prob"].idxmax()]
# ).reset_index(drop=True)

# # Calculate the overall mean of 'gt_correct'
# overall_accuracy = aggregated["gt_correct"].mean()
# print ("total number of trajs:", len(df))
# print("Overall Accuracy:", overall_accuracy)

# # if avg_yes_prob > 0.5, the predicted answer is YES (1), otherwise NO (0)
# df['predicted'] = (df['avg_yes_prob'] > 0.5).astype(int)
# # Ensure the ground truth is represented as 0 or 1 (if it's not already)
# df['gt_correct'] = df['gt_correct'].astype(int)
# # Create a column that shows if the prediction matches the ground truth for that entry
# df['correct_prediction'] = (df['predicted'] == df['gt_correct']).astype(int)

# # # Print per-entry details (you can adjust the columns as needed)
# # print("Per-entry accuracy details:")
# # print(df[['docker_images', 'avg_yes_prob', 'gt_correct', 'predicted', 'correct_prediction']])

# # Calculate the overall average accuracy
# overall_accuracy = df['correct_prediction'].mean()
# print("\nOverall RM Accuracy:", overall_accuracy)

# # Compute descriptive statistics for avg_yes_prob
# desc_stats = df['avg_yes_prob'].describe()
# print("Descriptive statistics for avg_yes_prob:")
# print(desc_stats)


# df = pd.concat([df1, df2, df3], ignore_index=True)

# df = pd.concat([df1, df2, df3], ignore_index=True)
# df = df2
# df = df[:2010]
# def main():
#     run_aggregation(df)

df = pd.read_csv("./deepswe-verifier-only-matching-pairs-qwen3-14b-lora64_N16_traindata_p2p-14B-75k-lr1en5-v1.csv")

def run_aggregation(df):
    # Group by 'docker_images' and select the row with the maximum avg_yes_prob from each group
    aggregated = df.groupby("docker_images", as_index=False).apply(
        lambda group: group.loc[group["avg_yes_prob"].idxmax()]
    ).reset_index(drop=True)

    # Calculate the overall mean of 'gt_correct'
    overall_accuracy = aggregated["gt_correct"].mean()
    print ("total number of trajs:", len(df))
    print("Overall Accuracy:", overall_accuracy)

    # if avg_yes_prob > 0.5, the predicted answer is YES (1), otherwise NO (0)
    df['predicted'] = (df['avg_yes_prob'] > 0.5).astype(int)
    df['gt_correct'] = df['gt_correct'].astype(int)
    df['correct_prediction'] = (df['predicted'] == df['gt_correct']).astype(int)

    # Calculate the overall average accuracy
    overall_accuracy = df['correct_prediction'].mean()
    print("\nOverall RM Accuracy:", overall_accuracy)


    k=10
    acc_topk = accuracy_topkyesprob_then_p2p(df,k=k)
    print(f"Accuracy (top k [{k}] yesprob -> highest p2p_rate):", acc_topk)

    k=1
    acc_topk = accuracy_topkp2p_then_yesprob(df,k=k)
    print(f"Accuracy (top k [{k}] p2p -> highest yesprob):", acc_topk)

run_aggregation(df)

# if __name__=='main':
#     main