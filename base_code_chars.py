
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

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load the pre-trained VGG16 model
vgg16 = models.vgg16(pretrained=True).features.to(device)

# Disable gradient computations (we won't be training VGG16)
for param in vgg16.parameters():
    param.requires_grad = False

# Transformation for the images
transform = transforms.Compose(
    [
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
vgg16 = models.vgg16(pretrained=True).features.to(device)
#vgg16 = torch.nn.Sequential(*list(vgg16.children())[:-1])

# A function to extract features for a given image
def extract_features(img_path):
    img = Image.open(img_path).convert("RGB")
    img_t = transform(img).unsqueeze(0).to(device)
    with torch.no_grad():
        features = vgg16(img_t)
    return features.cpu().numpy().flatten()


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

dataset_path = "images_background"
data_images_dict = extract_image_data(dataset_path, extract_features_fn=extract_features, early_stop=True)

# char_class_encoder = LabelEncoder()
# char_class_id_map = char_class_encoder.fit(list(data_images_dict.keys()))


def create_graph_from_dataset(dataset_path):
    characters = {}
    edges = []
    edge_attr = []

    # Label encoding for eras
    era_encoder = LabelEncoder()
    eras = ["01", "02", "03", "04", "05"]
    era_encoder.fit(eras)

    # Load and process each image
    for root, dirs, files in tqdm(os.walk(dataset_path)):
        for file in sorted(files):
            if file.endswith(".png") and file.split("_")[-1][:2] in eras:
                char_class, era = os.path.basename(root), file.split("_")[-1][:2]
                if char_class not in characters:
                    characters[char_class] = []

                img_path = os.path.join(root, file)
                features = extract_features(img_path)
                node_id = len(characters[char_class])
                characters[char_class].append((node_id, features, era))
                
                # Create edges between consecutive eras for the same character
                if node_id > 0:  # Avoid the first era as it has no previous era
                    src = characters[char_class][node_id - 1][0]
                    dst = node_id
                    edges.append((src, dst))
                    edge_attr.append(era_encoder.transform([era])[0])

    # Compile all node features and era labels
    all_features = []
    all_labels = []
    for char_class in characters:
        for node_id, features, era in characters[char_class]:
            all_features.append(features)
            all_labels.append(era_encoder.transform([era])[0])

    # Convert to tensors
    edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
    node_features = torch.tensor(all_features, dtype=torch.float)
    labels = torch.tensor(all_labels, dtype=torch.long)

    # Create PyG Data object
    data = Data(x=node_features, edge_index=edge_index, y=labels)
    return data






characters = {}
edges = []
edge_attr = []

# Load and process each image
for root, dirs, files in tqdm(os.walk(dataset_path)):
    for file in sorted(files):
        if file.endswith(".png") and file.split("_")[-1][:2] in eras:
            char_class, era = os.path.basename(root), file.split("_")[-1][:2]
            if char_class not in characters:
                characters[char_class] = []
            
            img_path = os.path.join(root, file)
            features = extract_features(img_path)
            characters[char_class].append((features, era, img_path))



edge_index = torch.tensor(edges, dtype=torch.long) 
node_features = torch.tensor(all_features, dtype=torch.float)
labels = torch.tensor(all_labels, dtype=torch.long)

data = Data(x=node_features, edge_index=edge_index.t().contiguous(), y = labels)


###------OK FROM HERE
device = 'cuda' if torch.cuda.is_available() else 'cpu'  # check if cuda is available to send the model and tensors to the GPU
model = Node2Vec(data.edge_index, embedding_dim=128, walk_length=20,
                 context_size=10, walks_per_node=10,
                 num_negative_samples=1, p=1, q=1, sparse=True).to(device)

loader = model.loader(batch_size=128, shuffle=True, num_workers=4)  # data loader to speed the train 
optimizer = torch.optim.SparseAdam(list(model.parameters()), lr=0.01)  # initzialize the optimizer 


def train():
    model.train()  # put model in train model
    total_loss = 0
    for pos_rw, neg_rw in tqdm(loader):
        optimizer.zero_grad()  # set the gradients to 0
        loss = model.loss(pos_rw.to(device), neg_rw.to(device))  # compute the loss for the batch
        loss.backward()
        optimizer.step()  # optimize the parameters
        total_loss += loss.item()
    return total_loss / len(loader)


for epoch in range(1, 100):
    loss = train()
    print(f'Epoch: {epoch:02d}, Loss: {loss:.4f}')

all_vectors = ""
for tensor in model(torch.arange(data.num_nodes, device=device)):
    s = "\t".join([str(value) for value in tensor.detach().cpu().numpy()])
    all_vectors += s + "\n"
# save the vectors
with open("vectors.txt", "w") as f:
    f.write(all_vectors)
# save the labels
with open("labels.txt", "w") as f:
    f.write("\n".join([str(label) for label in data.y.numpy()]))





## OLD CODE:

# Create a graph from the dataset



# Path to your 'images_background' directory
dataset_path = "images_background"
graph_data = create_graph_from_dataset(dataset_path)
print(graph_data)
