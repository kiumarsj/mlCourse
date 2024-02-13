import os
from io import BytesIO
from PIL import Image

from sqlalchemy import create_engine, Column, Integer, String, BLOB
from sqlalchemy.orm import DeclarativeBase
from sqlalchemy.orm import sessionmaker

DATABASE_FILE = "DB_of_images.db"
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
    label = Column(String)  # Add a new column for labels

# Create the table
Base.metadata.create_all(bind=engine)

# Read labels from Labels.txt
labels_file = "Labels.txt"
with open(labels_file, "r") as f:
    labels = f.read().splitlines()

# Insert data into the database with labels
Session = sessionmaker(bind=engine)
session = Session()

image_dir = 'img/'
image_paths = [os.path.join(image_dir, img) for img in os.listdir(image_dir)]

for i, (img_path, label) in enumerate(zip(image_paths, labels)):
    with open(img_path, "rb") as f:
        img_data = f.read()

    # Open the image using Pillow to extract image format and size information
    with Image.open(BytesIO(img_data)) as img:
        format_info = img.format
        size_info = img.size

    image = ImageTable(id=i, name=img_path, file=img_data, label=label)
    session.add(image)

session.commit()

# Load images from database
dataset = session.query(ImageTable).all()
session.close()
