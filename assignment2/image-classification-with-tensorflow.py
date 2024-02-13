import os
import numpy as np
from io import BytesIO
from PIL import Image
from keras.preprocessing import image as IMG
from keras.preprocessing.image import img_to_array
from keras.applications.inception_v3 import InceptionV3
from keras.applications.inception_v3 import preprocess_input
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

from sqlalchemy import create_engine, Column, Integer, String, BLOB
from sqlalchemy.orm import DeclarativeBase
from sqlalchemy.orm import sessionmaker

DATABASE_FILE = "db-tensorflow.db"
if os.path.exists(DATABASE_FILE):
    os.remove(DATABASE_FILE)
DATABASE_URL = f"sqlite:///{DATABASE_FILE}"
engine = create_engine(DATABASE_URL)
class Base(DeclarativeBase):
    pass

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

def natural_sort_key(s):
    import re
    return [int(text) if text.isdigit() else text.lower() for text in re.split('([0-9]+)', s)]

# Import images from the img/ directory to database
image_dir = 'img/'
# image_paths = [os.path.join(image_dir, img) for img in os.listdir(image_dir)]
image_paths = sorted([os.path.join(image_dir, img) for img in os.listdir(image_dir) if img.endswith(('.jpg', '.png', '.jpeg'))], key=natural_sort_key)
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

# Load images from database
images = session.query(ImageTable).all()
# images = [img.file for img in images]
session.close()

# ------------------------------------------------------------------------------------

# Load a pre-trained CNN model (InceptionV3 in this example)
base_model = InceptionV3(weights='imagenet', include_top=False)

def extract_features_from_bytes(img_bytes):
    # Convert image bytes to image object
    img = IMG.load_img(BytesIO(img_bytes), target_size=(299, 299))
    img_array = img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)
    features = base_model.predict(img_array)
    features = np.reshape(features, (features.shape[0], -1))
    return features

def load_images_features(image_objects):
    images = []
    for img_obj in image_objects:
        img_data = img_obj.file
        features = extract_features_from_bytes(img_data)
        images.append(features)
    return np.vstack(images)


def perform_clustering(features, num_clusters):
    # scaler = StandardScaler()
    # features_scaled = scaler.fit_transform(features)
    features_scaled = features

    # Print the shape of the scaled features
    print("Shape of scaled features:", features_scaled.shape)

    # Use the minimum of samples and features as the maximum number of components
    n_components = min(features_scaled.shape[0], features_scaled.shape[1])

    # Use a smaller value for n_components
    pca = PCA(n_components=min(n_components, 100))
    # pca = PCA(n_components=50)
    features_pca = pca.fit_transform(features_scaled)

    kmeans = KMeans(n_clusters=num_clusters, random_state=42, max_iter=500)
    # labels = kmeans.fit_predict(features_pca)
    for i in range(200):
        kmeans.fit(features_pca)
    labels = kmeans.predict(features_pca)
    return labels


def visualize_clusters(features, labels):
    pca = PCA(n_components=10)
    features_pca = pca.fit_transform(features)

    plt.scatter(features_pca[:, 0], features_pca[:, 1], c=labels, cmap='viridis', s=20)
    plt.title('Image Clustering')
    plt.show()

category_mapping = {
    0: 'human',
    1: 'animal',
    2: 'neither',
}

num_clusters = category_mapping.__len__()

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