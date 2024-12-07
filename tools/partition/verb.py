import numpy as np
import torch
import random
import json
import pickle
import pandas as pd
import matplotlib.pyplot as plt

import benepar, spacy
from click import password_option

from tqdm import tqdm

nlp = spacy.load('en_core_web_md')

nlp.add_pipe("benepar", config={"model": "benepar_en3"})
doc = nlp("The time for action is now. It's never too late to do something.")


def find_root_verb_and_its_dobj(tree_root):
    # first check if the current node and its children satisfy the condition
    if tree_root.pos_ == "VERB":
        for child in tree_root.children:
            if child.dep_ == "dobj" and child.pos_ == "NOUN":
                return tree_root.lemma_, child.lemma_
        return tree_root.lemma_, None
    # if not, check its children
    for child in tree_root.children:
        return find_root_verb_and_its_dobj(child)
    # if no children satisfy the condition, return None
    return None, None


def find_root_verb_and_its_dobj_in_string(s):
    doc = nlp(s)
    first_sent = list(doc.sents)[0]
    return find_root_verb_and_its_dobj(first_sent.root)


def find_verb(data):
    for idx, data_item in tqdm(enumerate(data)):
        # 调用find_root_verb_and_its_dobj_in_string函数找到动词和宾语
        try:
            verb, noun = find_root_verb_and_its_dobj_in_string(data_item['instruction'])

            data_item['verb'] = verb
        except:
            data_item['verb'] = None
    return data


def verb_visualize(
        data,
        save_path,
        threshold=0.05,
        row=1,
        col=1
):
    if row * col > 1:
        fig, axes = plt.subplots(row, col, figsize=(col * 5, row * 5))
        ax = axes.flatten()
        for idx, item in enumerate(data):
            data_size = len(item)
            df = pd.DataFrame(item)
            df = df['verb'].value_counts()
            df = df[df > data_size * threshold]
            df.plot(kind='pie', ax=ax[idx], autopct='%1.1f%%', title=f"Client {idx}")
    else:
        data_size = len(data)
        fig, ax = plt.subplots(figsize=(10, 10))
        df = pd.DataFrame(data)
        df = df['verb'].value_counts()
        df = df[df > data_size * threshold]
        df.plot(kind='pie', ax=ax, autopct='%1.1f%%')
    plt.savefig(save_path)


def dirichlet_partition(
        data_path,
        save_path,
        num_clients,
        train_frac=0.8,
        val_frac=0.1,
        alpha=1.0,
        client_sample_nums=None,
        verbose=False,
        seed=0
):
    dataset = {
        "train": [],
        "valid": [],
        "test": [],
        f"clients={num_clients}_alpha={alpha}": {
            "attribute": {"clients_num": num_clients, "alpha": alpha},
            "train": {idx:[] for idx in range(num_clients)},
            "valid": {idx:[] for idx in range(num_clients)},
            "test": {idx:[] for idx in range(num_clients)}
        }
    }
    with open(data_path, "r") as file:
        raw = json.load(file)
    if 'verb' not in raw[0].keys():
        print("No 'verb' key in data, start finding verb in data...")
        raw = find_verb(raw)
        print("Finding verb in data finished.")
        with open(data_path.replace(".pkl", "_verb.pkl"), 'w') as f:
            json.dump(raw, f, indent=4)
    if client_sample_nums is None:
        data_size = len(raw)
        client_sample_nums = [data_size // num_clients] * num_clients
        for i in range(data_size % num_clients):
            client_sample_nums[i] += 1
    verb_map = {}
    idx_list = []
    for i, item in enumerate(raw):
        if item['verb'] not in verb_map:
            verb_map[item['verb']] = len(verb_map)
            idx_list.append([])
        idx_list[verb_map[item['verb']]].append(i)

    verb_visualize(raw, save_path=save_path.replace(".pkl", ".svg"))

    num_classes = len(verb_map)
    class_amount = [len(idx_list[i]) for i in range(num_classes)]

    class_priors = np.random.dirichlet(alpha=[alpha] * num_classes,
                                       size=num_clients)
    prior_cumsum = np.cumsum(class_priors, axis=1)

    client_indices = [np.zeros(client_sample_nums[cid]).astype(np.int64) for cid in
                      range(num_clients)]

    while np.sum(client_sample_nums) != 0:
        curr_cid = np.random.randint(num_clients)
        # If current node is full resample a client
        if verbose:
            print('Remaining Data: %d' % np.sum(client_sample_nums))
        if client_sample_nums[curr_cid] <= 0:
            continue
        client_sample_nums[curr_cid] -= 1
        curr_prior = prior_cumsum[curr_cid]
        while True:
            curr_class = np.argmax(np.random.uniform() <= curr_prior)
            # Redraw class label if no rest in current class samples
            if class_amount[curr_class] <= 0:
                continue
            class_amount[curr_class] -= 1
            client_indices[curr_cid][client_sample_nums[curr_cid]] = \
                idx_list[curr_class][class_amount[curr_class]]

            break

    client_dict = {cid: client_indices[cid] for cid in range(num_clients)}
    client_data = [[] for _ in range(num_clients)]
    for cid in range(num_clients):
        for idx in client_dict[cid]:
            raw[idx]['response'] = raw[idx]['output']
            client_data[cid].append(raw[idx])

    verb_visualize(client_data, save_path=save_path.replace(".pkl", "_partition.svg"), row=2, col=5)

    for i in range(num_clients):
        indexes = np.arange(len(client_data[i]))
        np.random.seed(seed)
        np.random.shuffle(indexes)
        train_index = indexes[:int(len(client_data[i]) * train_frac)]
        valid_index = indexes[int(len(client_data[i]) * train_frac): int(len(client_data[i]) * (train_frac + val_frac))]
        test_index = indexes[int(len(client_data[i]) * (train_frac + val_frac)):]
        for idx in train_index:
            dataset["train"].append(client_data[i][idx])
            dataset[f"clients={num_clients}_alpha={alpha}"]["train"][i].append(len(dataset["train"]) - 1)
        for idx in valid_index:
            dataset["valid"].append(client_data[i][idx])
            dataset[f"clients={num_clients}_alpha={alpha}"]["valid"][i].append(len(dataset["valid"]) - 1)
        for idx in test_index:
            dataset["test"].append(client_data[i][idx])
            dataset[f"clients={num_clients}_alpha={alpha}"]["test"][i].append(len(dataset["test"]) - 1)
    with open(save_path, "wb") as file:
        pickle.dump(dataset, file)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, required=True)
    parser.add_argument("--save_path", type=str, required=True)
    parser.add_argument("--num_clients", type=int, default=10)
    parser.add_argument("--alpha", type=float, default=1.0)
    parser.add_argument("--train_frac", type=float, default=0.8)
    parser.add_argument("--val_frac", type=float, default=0.1)
    parser.add_argument("--client_sample_nums", type=list, default=None)
    args = parser.parse_args()
    dirichlet_partition(
        args.data_path,
        args.save_path,
        args.num_clients,
        args.train_frac,
        args.val_frac,
        args.alpha,
        args.client_sample_nums
    )

# python verb.py --data_path ../data/alpaca_gpt4_data_cherry_verb.json --save_path ../data/alpaca_gpt4_data_verb_partition.pkl --num_clients 10 --train_frac 0.8 --val_frac 0.1 --alpha 1.0