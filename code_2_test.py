# %%

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
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import random
import siamese
os.environ['CUDA_LAUNCH_BLOCKING'] = '0'


#import torch_cluster

# %%
# Transformation for the images
transform = transforms.Compose(
    [
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)


# %%
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
vgg16 = models.vgg16(pretrained=True).features.to(device)
#vgg16 = torch.nn.Sequential(*list(vgg16.children())[:-1])

# %%

# A function to extract features for a given image
def extract_features(img_path):
    img = Image.open(img_path).convert("RGB")
    img_t = transform(img).unsqueeze(0).to(device)
    with torch.no_grad():
        features = vgg16(img_t)
    return features.cpu().numpy().flatten()



# %%

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
        if t > 200 and early_stop:
            break
    return characters



# %%
dataset_path = "images_background"
data_images_dict = extract_image_data(dataset_path, extract_features_fn=extract_features, early_stop=False)

# %%
char_class_encoder = LabelEncoder()
char_class_id_map = char_class_encoder.fit(list(data_images_dict.keys()))


# %%
def generate_graph_data_from_dict(data_images_dict, gamma=0.0075):
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
data = generate_graph_data_from_dict(data_images_dict)
if data.validate(raise_on_error=True):
    print("DATA VALIDATED")
    print(f"NUM NODES: {data.num_nodes}")
    print(f"NUM EDGES: {data.num_edges}")

# %%
#print("The object Data is")
#print(data.edge_index)

# %%
def load_graph_from_data(data):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'  # check if cuda is available to send the model and tensors to the GPU
    model = Node2Vec(data.edge_index, embedding_dim=128, walk_length=30,
                    context_size=10, walks_per_node=10,
                    num_negative_samples=1, p=1, q=0.8, sparse=True).to(device)
    loader = model.loader(batch_size=128, shuffle=True, num_workers=4)  # data loader to speed the train 
    optimizer = torch.optim.SparseAdam(list(model.parameters()), lr=0.01)  # initzialize the optimizer 
    return model, loader, optimizer

# %%

model, loader, optimizer = load_graph_from_data(data)
print("Model.forward is: "+ str(model.forward))

def train():
    model.train()  # put model in train model
    total_loss = 0
    for pos_rw, neg_rw in loader:
        optimizer.zero_grad()  # set the gradients to 0
        loss = model.loss(pos_rw.to(device), neg_rw.to(device))  # compute the loss for the batch
        loss.backward()
        optimizer.step()  # optimize the parameters
        total_loss += loss.item()
    return total_loss / len(loader)

for epoch in tqdm(range(1, 15)):
    loss = train()
    print(f'Epoch: {epoch:02d}, Loss: {loss:.4f}')

all_vectors = ""
for tensor in model(torch.arange(data.num_nodes, device=device)):
    # print(f"TENSOR IS: {tensor}")
    s = "\t".join([str(value) for value in tensor.detach().cpu().numpy()])
    all_vectors += s + "\n"
# save the vectors
with open("vectors.txt", "w") as f:
    f.write(all_vectors)
# save the labels
with open("labels.txt", "w") as f:
    f.write("\n".join([str(label) for label in data.y.numpy()]))


# Convert string representations of tensors to actual tensors
vector_data = np.loadtxt("vectors.txt")
label_data = np.loadtxt("labels.txt")

# Perform t-SNE dimensionality reduction
tsne = TSNE(n_components=2, verbose=1, perplexity=40, n_iter=300)
tsne_results = tsne.fit_transform(vector_data)

# Plot the results with labels colored
plt.figure(figsize=(16,10))
scatter = plt.scatter(tsne_results[:,0], tsne_results[:,1], c=label_data, cmap='viridis', alpha=0.5)
plt.colorbar(scatter)
plt.title('t-SNE projection of the character vectors')
plt.xlabel('Component 1')
plt.ylabel('Component 2')
plt.savefig('/n/home02/rskeweszorrilla/chinese_char_evolution/graph_6pm.png')


# %%
