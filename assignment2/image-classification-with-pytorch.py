import os
import numpy as np
from io import BytesIO
from PIL import Image
import torch
from torchvision import models, transforms

from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

from sqlalchemy import create_engine, Column, Integer, String, BLOB
from sqlalchemy.orm import declarative_base, sessionmaker

DATABASE_FILE = "db-pytorch.db"
if os.path.exists(DATABASE_FILE):
    os.remove(DATABASE_FILE)
DATABASE_URL = f"sqlite:///{DATABASE_FILE}"
engine = create_engine(DATABASE_URL)
Base = declarative_base()

class ImageTable(Base):
    __tablename__ = "image"
    id = Column(Integer, primary_key=True, index=True)
    name = Column(String)
    file = Column(BLOB)
    cluster = Column(String)

# Create the table
Base.metadata.create_all(bind=engine)

# Insert data into the database
Session = sessionmaker(bind=engine)
session = Session()

# Import images from the img/ directory to the database
image_dir = 'img/'
image_paths = [os.path.join(image_dir, img) for img in os.listdir(image_dir)]
for i, img_path in enumerate(image_paths):
    with open(img_path, "rb") as f:
        img_data = f.read()

    # Open the image using Pillow to extract image format and size information
    with Image.open(BytesIO(img_data)) as img:
        format_info = img.format
        size_info = img.size

    image = ImageTable(id=i, name=img_path, file=img_data)
    session.add(image)

session.commit()

# Load images from the database
images = session.query(ImageTable).all()
session.close()

# ------------------------------------------------------------------------------------

# Load a pre-trained PyTorch model (ResNet50 in this example)
model = models.resnet50(pretrained=True)
model = model.eval()

def extract_features_from_bytes(img_bytes):
    # Convert image bytes to a PyTorch tensor
    img = Image.open(BytesIO(img_bytes)).convert('RGB')
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    img_tensor = transform(img).unsqueeze(0)

    # Extract features using the model
    with torch.no_grad():
        features = model(img_tensor)

    # Reshape the features
    features = features.squeeze().numpy()
    return features

def load_images_features(image_objects):
    images = []
    for img_obj in image_objects:
        img_data = img_obj.file
        features = extract_features_from_bytes(img_data)
        images.append(features)
    return np.vstack(images)

def perform_clustering(features, num_clusters):
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)

    # Print the shape of the scaled features
    print("Shape of scaled features:", features_scaled.shape)

    # Use the minimum of samples and features as the maximum number of components
    n_components = min(features_scaled.shape[0], features_scaled.shape[1])

    # Use a smaller value for n_components
    pca = PCA(n_components=min(n_components, 10))
    features_pca = pca.fit_transform(features_scaled)

    kmeans = KMeans(n_clusters=num_clusters, random_state=42)
    labels = kmeans.fit_predict(features_pca)
    return labels

def visualize_clusters(features, labels):
    pca = PCA(n_components=2)
    features_pca = pca.fit_transform(features)

    plt.scatter(features_pca[:, 0], features_pca[:, 1], c=labels, cmap='viridis', s=20)
    plt.title('Image Clustering')
    plt.show()

category_mapping = {
    0: 'neither',
    1: 'human',
    2: 'animal',
}

num_clusters = len(category_mapping)

# Load and preprocess images
image_features = load_images_features(images)

# Perform clustering
cluster_labels = perform_clustering(image_features, num_clusters)

# Update the database with cluster labels
for i, label in enumerate(cluster_labels):
    image = session.query(ImageTable).filter_by(id=i).first()
    if image:
        category = category_mapping.get(label, 'unknown')
        image.cluster = category

session.commit()

# Load and print images from the database with updated cluster labels
images = session.query(ImageTable).all()
for img in images:
    print(f"ID: {img.id}, Name: {img.name}, Cluster: {img.cluster}")

# Visualize clusters
visualize_clusters(image_features, cluster_labels)
