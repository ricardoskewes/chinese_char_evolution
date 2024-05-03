import torch
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
import os
import numpy as np
from sklearn.preprocessing import LabelEncoder
from torch_geometric.data import Data
from tqdm import tqdm

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


# A function to extract features for a given image
def extract_features(img_path):
    img = Image.open(img_path).convert("RGB")
    img_t = transform(img).unsqueeze(0).to(device)
    with torch.no_grad():
        features = vgg16(img_t)
    return features.cpu().numpy().flatten()


# Create a graph from the dataset
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


# Path to your 'images_background' directory
dataset_path = "images_background"
graph_data = create_graph_from_dataset(dataset_path)
print(graph_data)
