import argparse
import csv
import ast
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter

# Set up argument parser to accept multiple result files
parser = argparse.ArgumentParser(description="Analyse multiple evaluation result files")
parser.add_argument(
    "results_files", nargs="+", type=str, help="List of result files to analyse"
)
args = parser.parse_args()

colors = ["blue", "red", "green", "purple", "orange", "brown"]  # For multiple datasets
plt.figure(figsize=(8, 5))  # Initialize the figure

# Loop through each file provided in command line arguments
for file_idx, results_file in enumerate(args.results_files):
    data = []

    # Read CSV file
    with open(results_file, "r") as f:
        reader = csv.reader(f)
        for row in reader:
            data.append(row)

    # Extract basic information
    global_highest_stage = data[0][1]
    average_stage = data[1][1]

    print(f"File: {results_file}")
    print("Global highest stage reached:", global_highest_stage)
    print("Average stage reached:", average_stage)

    headers = data[2]
    iters = data[3:]

    rews = []
    stages = []

    # Extract rewards
    for iter in iters:
        highest_stage = iter[1]
        stages.append(highest_stage)
        rew_over_time = ast.literal_eval(iter[2])
        rews.append(rew_over_time)

    # Find max rollout length for padding
    max_length = max(len(r) for r in rews)

    # Pad shorter rollouts with NaN
    padded_rewards = np.full((len(rews), max_length), np.nan)
    for i, r in enumerate(rews):
        padded_rewards[i, : len(r)] = r  # Fill available data

    # Compute mean and standard deviation, ignoring NaNs
    mean_rewards = np.nanmean(padded_rewards, axis=0)
    std_rewards = np.nanstd(padded_rewards, axis=0)

    # Timesteps
    timesteps = np.arange(max_length)

    # Select color for this dataset
    color = colors[file_idx % len(colors)]

    # Plot mean and std deviation for this dataset
    plt.plot(timesteps, mean_rewards, label=f"Mean ({results_file})", color=color)
    plt.fill_between(
        timesteps,
        mean_rewards - std_rewards,
        mean_rewards + std_rewards,
        color=color,
        alpha=0.2,
        label=f"Std Dev ({results_file})",
    )

# Labels and title
plt.xlabel("Environment Steps")
plt.ylabel("Cumulative Reward")
plt.title("RL Policy Rollouts - Cumulative Reward over Time")
plt.legend(loc="upper left")

# Show final overlay plot
plt.show()
