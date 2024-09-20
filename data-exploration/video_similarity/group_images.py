import os

import numpy as np
import cv2
from transformers import AutoFeatureExtractor, AutoModel
import torch
import torchvision.transforms as T
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
from PIL import Image


class Node:
    def __init__(self, prefix, count, parrent_prefix=""):
        self.prefix = prefix
        self.parrent_prefix = parrent_prefix
        self.leafs = []
        self.count = count

        self.full_name = f"{self.parrent_prefix}_{self.prefix}" if self.parrent_prefix else self.prefix

    def __str__(self):
        return f"'{self.full_name}': {self.count}"

def build_tree(filenames, prefix="", parrent_prefix=""):
    node = Node(prefix, len(filenames), parrent_prefix)
    leafs = []

    if len(filenames) == 1 and filenames[0] == "":
        return node

    filename_parts = [filename.split("_", 1) for filename in filenames]
    prefixes, suffixes= zip(*[i if len(i) > 1 else i + [""] for i in filename_parts])

    filename_parts = [filename.split("_", 1) for filename in filenames]
    prefixes, suffixes = zip(*[i if len(i) > 1 else i + [""] for i in filename_parts])

    uprefixes, inverse_idxs = np.unique(prefixes, return_inverse=True)

    for idx, uprefix in enumerate(uprefixes):
        leaf = build_tree(np.array(suffixes)[inverse_idxs == idx], uprefix, node.full_name)
        leafs.append(leaf)

    node.leafs = leafs
    return node


def print_tree(node, level=0, is_last=False, prefix=''):
    if level == 0:
        print(node)
    else:
        print(prefix, '└───' if is_last else '├───', node, sep='')
    prefix += '    ' if is_last else '│   '
    for i, leaf in enumerate(node.leafs):
        print_tree(leaf, level + 1, i == len(node.leafs) - 1, prefix)


def get_all_leafs(node):
    if len(node.leafs) == 0:
        return [node]
    else:
        leafs = []
        for leaf in node.leafs:
            leafs += get_all_leafs(leaf)
    return leafs


def get_all_leaf_full_names(node):
    return [leaf.full_name for leaf in get_all_leafs(node)]


def is_next_level_has_the_same_count(node):
    return len(node.leafs) == 1 and node.count == node.leafs[0].count


def split_by_groups(node, groups, level=0):
    if level >= 5:
        groups[node.full_name] = get_all_leaf_full_names(node)
    elif sum([leaf.count for leaf in node.leafs]) == len(node.leafs):
        groups[node.full_name] = get_all_leaf_full_names(node)
    elif is_next_level_has_the_same_count(node):
        groups[node.full_name] = get_all_leaf_full_names(node)
    else:
        for leaf in node.leafs:
            split_by_groups(leaf, groups, level + 1)


class Embedder:
    def __init__(self) -> None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print("Device:", device)

        model_ckpt = "google/vit-base-patch16-224-in21k"
        extractor = AutoFeatureExtractor.from_pretrained(model_ckpt)
        self.model = AutoModel.from_pretrained(model_ckpt).to(device)

        # Data transformation chain.
        self.transformation_chain = T.Compose([
            # We first resize the input image to 256x256 and then we take center crop.
            T.Resize((extractor.size["width"], extractor.size["height"])),
            T.ToTensor(),
            T.Normalize(mean=extractor.image_mean, std=extractor.image_std),
        ])

    def extract_embedding_from_img(self, img: np.ndarray):
        """Utility to extract embeddings from a single image."""
        device = self.model.device
        img = Image.fromarray(img)
        img = self.transformation_chain(img)
        img = img.unsqueeze(0).to(device)
        with torch.no_grad():
            embeddings = self.model(pixel_values=img).last_hidden_state[:, 0].cpu()
        return embeddings

    @staticmethod
    def compute_scores(emb_one, emb_two):
        """Computes cosine similarity between two vectors."""
        scores = torch.nn.functional.cosine_similarity(emb_one, emb_two)
        return scores.numpy().tolist()


def img_by_name(img_name: str, df: pd.DataFrame, base_pth: str, img_prefix_folder: str = "") -> np.ndarray:
    row = df[df["name"] == img_name]
    pth = os.path.join(base_pth, row["batch"].iloc[0], str(row["folder"].iloc[0]), img_prefix_folder, img_name)
    img = cv2.cvtColor(cv2.imread(pth), cv2.COLOR_BGR2RGB)
    return img


def dfs(node, visited, M, component):
    visited[node] = True
    component.append(node)
    for i, val in enumerate(M[node]):
        if val == 1 and not visited[i]:
            dfs(i, visited, M, component)

def find_connected_components(M):
    visited = [False] * len(M)
    components = []
    for i in range(len(M)):
        if not visited[i]:
            component = []
            dfs(i, visited, M, component)
            components.append(component)
    return components


def get_elements(x, row_indices, col_indices):
    return np.array([x[i, j] for i in row_indices for j in col_indices])


def compute_group_similarities(names: list[str], groups: dict, group_names: dict, sim_mtx: np.ndarray) -> tuple:
    avg_inner_group_similarities = []
    for i in tqdm(range(len(group_names)), desc="Computing inner group similarities"):
        indices0 = np.where(np.isin(names, groups[group_names[i]]))[0]
        avg_inner_group_similarities.append(np.mean(get_elements(sim_mtx, indices0, indices0)))

    avg_outer_goup_similarities = np.zeros((len(group_names), len(group_names)))
    max_outer_group_similarities = np.zeros((len(group_names), len(group_names)))
    for i in tqdm(range(len(group_names) - 1), desc="Computing outer group similarities"):
        for j in range(i + 1, len(group_names)):
            indices0 = np.where(np.isin(names, groups[group_names[i]]))[0]
            indices1 = np.where(np.isin(names, groups[group_names[j]]))[0]

            sims = get_elements(sim_mtx, indices0, indices1)
            avg_outer_goup_similarities[i, j] = avg_outer_goup_similarities[j, i] = np.mean(sims)
            max_outer_group_similarities[i, j] = max_outer_group_similarities[j, i] = np.max(sims)

    return avg_inner_group_similarities, avg_outer_goup_similarities, max_outer_group_similarities


def draw_group(
        df: pd.DataFrame, base_pth: str, group_items: list, group_name: str, img_folder_prefix: str = "", n: int = 3
) -> None:
    if len(group_items) < n:
        subgroup = group_items
    else:
        subgroup = np.random.choice(group_items, n, replace=False)
    plt.figure(figsize=(15, 4))
    plt.suptitle(group_name, fontsize=14)
    for i, item in enumerate(subgroup):
        row = df[df["name"] == item]
        pth = os.path.join(base_pth, row["batch"].iloc[0], str(row["folder"].iloc[0]), img_folder_prefix, item)
        img = cv2.cvtColor(cv2.imread(pth), cv2.COLOR_BGR2RGB)
        plt.subplot(1, len(subgroup), i + 1)
        plt.imshow(img)
        plt.title(item)
