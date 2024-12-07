import json
import os.path
import pickle
import sys
sys.path.append(os.path.abspath('..'))
import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
from transformers import BitsAndBytesConfig, HfArgumentParser, AutoModelForCausalLM, AutoTokenizer
from selection.cherry.utils import DataAnalysisArguments
from selection.cherry.data_analysis import main as data_analysis_main


def kmeans_partition(data, num_clusters, random_state=0):
    """Partition data using kmeans.

    Args:
        data (np.ndarray): data to be partitioned.
        num_clusters (int): number of clusters.
        random_state (int, optional): random seed. Defaults

    Returns:
        (np.ndarray): cluster assignment.
    """
    kmeans = KMeans(n_clusters=num_clusters, random_state=random_state).fit(data)
    return kmeans.labels_


def do_reduce_dim(high_dim_vectors):
    # Perform t-SNE for visualization
    tsne = TSNE(n_components=2, random_state=0)
    low_dim_vectors = tsne.fit_transform(high_dim_vectors)
    return low_dim_vectors


def partition(
        data_path,
        model_path,
        embedding_path,
        save_path,
        num_clusters=10,
        train_frac=0.8,
        val_frac=0.1,
        alpha=0,
        seed=0
):
    dataset = {
        "train": [],
        "valid": [],
        "test": [],
        f"clients={num_clusters}_alpha={alpha}": {
            "attribute": {"clients_num": num_clusters, "alpha": alpha},
            "train": {idx:[] for idx in range(num_clusters)},
            "valid": {idx:[] for idx in range(num_clusters)},
            "test": {idx:[] for idx in range(num_clusters)}
        }
    }
    if not os.path.isfile(embedding_path):
        print("Embedding file not found, generating embeddings...")
        arg = DataAnalysisArguments(
            data_path=data_path,
            save_path=embedding_path,
            model_name_or_path=model_path,
            max_length=512,
            prompt="alpaca",
            mod="pre",
            quant=32
        )
        # arg = HfArgumentParser(DataAnalysisArguments).parse_args_into_dataclasses()[0]
        data_analysis_main(arg)
        print("Embedding generation done.")
    with open(data_path, "r") as file:
        raw = json.load(file)
    embedding_data = torch.load(embedding_path)
    embedding_data = np.array([item["sent_emb"][0].reshape(-1).numpy() for item in embedding_data])
    clusters = kmeans_partition(embedding_data, num_clusters, seed)
    low_dim_vectors = do_reduce_dim(embedding_data)
    cluster_idx = [[] for _ in range(num_clusters)]
    cluster_data = [[] for _ in range(num_clusters)]
    for i, item in enumerate(raw):
        item['response'] = item['output']
        cluster_data[clusters[i]].append(item)
        cluster_idx[clusters[i]].append(i)
    fig, ax = plt.subplots(figsize=(num_clusters * 1.5, num_clusters * 1.5))
    for i in range(num_clusters):
        ax.scatter(low_dim_vectors[cluster_idx[i], 0], low_dim_vectors[cluster_idx[i], 1], label=f"Cluster {i}")
    ax.legend()
    plt.savefig("tsne.svg")

    for i in range(num_clusters):
        indexes = np.arange(len(cluster_data[i]))
        np.random.seed(seed)
        np.random.shuffle(indexes)
        train_index = indexes[:int(len(cluster_data[i]) * train_frac)]
        valid_index = indexes[int(len(cluster_data[i]) * train_frac): int(len(cluster_data[i]) * (train_frac + val_frac))]
        test_index = indexes[int(len(cluster_data[i]) * (train_frac + val_frac)):]
        for idx in train_index:
            dataset["train"].append(cluster_data[i][idx])
            dataset[f"clients={num_clusters}_alpha={alpha}"]["train"][i].append(len(dataset["train"]) - 1)
        for idx in valid_index:
            dataset["valid"].append(cluster_data[i][idx])
            dataset[f"clients={num_clusters}_alpha={alpha}"]["valid"][i].append(len(dataset["valid"]) - 1)
        for idx in test_index:
            dataset["test"].append(cluster_data[i][idx])
            dataset[f"clients={num_clusters}_alpha={alpha}"]["test"][i].append(len(dataset["test"]) - 1)
    with open(save_path, "wb") as file:
        pickle.dump(dataset, file)

# acl ijcai
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, required=True)
    parser.add_argument("--embedding_path", type=str, required=True)
    parser.add_argument("--model_path", type=str, default=None)
    parser.add_argument("--save_path", type=str, required=True)
    parser.add_argument("--num_clusters", type=int, default=10)
    parser.add_argument("--train_frac", type=float, default=0.8)
    parser.add_argument("--val_frac", type=float, default=0.1)
    parser.add_argument("--alpha", type=float, default=0)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    partition(
        args.data_path,
        args.model_path,
        args.embedding_path,
        args.save_path,
        args.num_clusters,
        args.train_frac,
        args.val_frac,
        args.alpha,
        args.seed
    )
# python kmeans.py --data_path ../data/alpaca_gpt4_data_cherry.json --embedding_path ../data/alpaca_gpt4_data_cherry.pt --save_path ../data/alpaca_gpt4_data_cherry_partition.pkl --num_clusters 10 --train_frac 0.8 --val_frac 0.1 --alpha 0 --seed 0