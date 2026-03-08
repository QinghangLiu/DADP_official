import json
import os
import argparse
import matplotlib.pyplot as plt
import numpy as np

def plot_distance_matrix(distance_json, output_path=None):
    with open(distance_json, "r") as f:
        data = json.load(f)
    pairwise = data["pairwise_mse"]
    task_ids = data["task_ids"]
    n = len(task_ids)
    # Build distance matrix
    dist_matrix = np.zeros((n, n))
    id_to_idx = {tid: i for i, tid in enumerate(task_ids)}
    for entry in pairwise:
        i = id_to_idx[entry["task_i"]]
        j = id_to_idx[entry["task_j"]]
        dist_matrix[i, j] = entry["mse"]
        dist_matrix[j, i] = entry["mse"]
    # Plot
    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(dist_matrix, cmap="viridis")
    ax.set_xticks(np.arange(n))
    ax.set_yticks(np.arange(n))
    ax.set_xticklabels(task_ids)
    ax.set_yticklabels(task_ids)
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    ax.set_xlabel("Task ID")
    ax.set_ylabel("Task ID")
    ax.set_title("Pairwise Embedding MSE Distance Between Tasks")
    fig.colorbar(im, ax=ax)
    plt.tight_layout()
    if output_path:
        plt.savefig(output_path)
        print(f"Saved plot to {output_path}")
    else:
        plt.show()

def main():
    parser = argparse.ArgumentParser(description="Plot embedding distance matrix")
    parser.add_argument("--distance_json", type=str, default="results/embedding_distance.json", help="Path to embedding distance json file")
    parser.add_argument("--output", type=str, default="results/figure", help="Path to save the plot (png)")
    args = parser.parse_args()
    plot_distance_matrix(args.distance_json, args.output)

if __name__ == "__main__":
    main()
