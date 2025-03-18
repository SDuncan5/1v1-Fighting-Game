import argparse
import csv
import ast
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter

parser = argparse.ArgumentParser(
    description="Analyse the results of the super cool evaluation"
)
parser.add_argument(
    "results_file", type=str, help="The file containing the results of the evaluation"
)
args = parser.parse_args()

data = []

with open(args.results_file, "r") as f:
    reader = csv.reader(f)
    for row in reader:
        data.append(row)

global_higest_stage = data[0][1]
average_stage = data[1][1]

print("global highest stage reached: ", global_higest_stage)
print("average stage reached: ", average_stage)

headers = data[2]
iters = data[3:]

rews = []
stages = []

# Plot setup
plt.figure(figsize=(8, 5))

for iter in iters:
    iter_num = iter[0]
    highest_stage = iter[1]
    stages.append(highest_stage)
    rew_over_time = ast.literal_eval(iter[2])
    rews.append(rew_over_time)

    plt.plot(rew_over_time)

plt.title("RL Policy Rollouts - Cumulative Reward Over Time")
plt.ylabel("Cumulative Reward")
plt.xlabel("Environment Steps")
plt.show()

# Find the maximum length of any rollout
max_length = max(len(r) for r in rews)

# Pad shorter rollouts with NaN to match the max length
padded_rewards = np.full((len(rews), max_length), np.nan)
for i, r in enumerate(rews):
    padded_rewards[i, : len(r)] = r  # Fill available data

# Compute mean and standard deviation, ignoring NaNs
mean_rewards = np.nanmean(padded_rewards, axis=0)
std_rewards = np.nanstd(padded_rewards, axis=0)

# Timesteps (assumes all rollouts start at timestep 0)
timesteps = np.arange(max_length)

# Plot setup
plt.figure(figsize=(8, 5))

# Plot mean cumulative reward with shaded region for std deviation
plt.plot(timesteps, mean_rewards, label="Mean Reward", color="blue")
plt.fill_between(
    timesteps,
    mean_rewards - std_rewards,
    mean_rewards + std_rewards,
    color="blue",
    alpha=0.2,
    label="Std Dev",
)

# Labels and title
plt.xlabel("Environment Steps")
plt.ylabel("Cumulative Reward")
plt.title("RL Policy Rollouts - Cumulative Reward Over Time")
plt.legend()

# Show plot
plt.show()


# Plot setup
plt.figure(figsize=(8, 5))

# Count occurrences of each unique value
value_counts = Counter(stages)

# Sort values by key (value itself)
sorted_values = sorted(value_counts.items())  # (value, frequency) pairs sorted by value

# Separate into lists for plotting
x_values, y_frequencies = zip(*sorted_values)

# Create bar plot
plt.bar(x_values, y_frequencies, color="blue", edgecolor="black", alpha=0.7)

# Labels and title
plt.xlabel("Stage Reached")
plt.ylabel("Frequency")
plt.title("Frequency of Stages Reached")

# Show plot
plt.show()
