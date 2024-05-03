#import networkx as nx
import matplotlib.pyplot as plt
import random as random
from tqdm import tqdm
import torch
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
import os
import numpy as np
from sklearn.preprocessing import LabelEncoder
from torch_geometric.data import Data
from tqdm import tqdm
from torch_geometric.nn import Node2Vec
from itertools import permutations
import torch_cluster


def extract_image_data(dataset_path, extract_features_fn = None, considered_eras = ['00', '01', '02', '03', '04', '05', '06'], early_stop= False):
    characters = {}
    edges = []
    edge_attr = []
    # Load and process each image
    t = 0
    for root, dirs, files in tqdm(os.walk(dataset_path)):
        for file in sorted(files):
            if file.endswith(".png") and file.split("_")[-1][:2] in considered_eras:
                char_class, era = os.path.basename(root), file.split("_")[-1][:2]
                if char_class not in characters:
                    characters[char_class] = []
                
                img_path = os.path.join(root, file)
                if extract_features_fn is not None:
                    features = extract_features_fn(img_path)
                    res = (t, features, era, img_path)
                else:
                    res = (t, None, era, img_path)

                characters[char_class].append(res)
                t+=1
        if t > 40 and early_stop:
            break
    return characters


data_images_dict = extract_image_data(dataset_path, extract_features_fn=extract_features, early_stop=False)

def generate_graph_data_from_dict(data_images_dict, gamma=0.1):
    edges = []
    X = []
    y = []
    labels = []  # This will collect labels for all nodes
    
    # Iterate through each character class and its data
    for char_class, data in data_images_dict.items():
        node_ids = [data[i][0] for i in range(len(data))]
        features = [data[i][1] for i in range(len(data))]
        eras = [data[i][2] for i in range(len(data))]

        # Append features and labels
        X += features
        y += eras
        labels += eras  # Collect labels for all nodes

        # Compute probabilistic edges between all node pairs within the same character class
        for i in range(len(node_ids)):
            for j in range(i + 1, len(node_ids)):  # Ensure we only calculate each pair once
                distance = np.linalg.norm(features[i] - features[j])
                probability = np.exp(-gamma * distance)
                if random.random() < probability:
                    edges.append((node_ids[i], node_ids[j]))
                    edges.append((node_ids[j], node_ids[i]))  # Since the graph is undirected

    # Convert all collected data into torch tensors
    E = torch.tensor(edges, dtype=torch.long).t().contiguous()
    X = torch.tensor(X, dtype=torch.float)
    y = torch.tensor([int(label) for label in labels], dtype=torch.long)  # Convert all labels to integer

    # Create PyG Data object
    data = Data(x=X, edge_index=E, y=y)
    return data


# Generate data using the modified function
data = generate_graph_data_from_dict(data_images_dict, gamma=0.1)
# Plot graph statistics
print("Nodes;")
print(data.num_nodes)
print("features;")
print(data.num_features)
