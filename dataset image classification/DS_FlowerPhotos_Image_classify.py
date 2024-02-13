import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from keras import layers
from keras.models import Sequential
print(f"Tensorflow version => {tf.__version__}")
import pathlib
import PIL


# dataset of about 3,700 photos of flowers:
#         1: daisy/
#         2: dandelion/
#         3: roses/
#         4: sunflowers/
#         5: tulips/

print("flower_photos dataset")
dataset_url = "https://storage.googleapis.com/download.tensorflow.org/example_images/flower_photos.tgz"
data_dir = tf.keras.utils.get_file('flower_photos.tar', origin=dataset_url, extract=True)
data_dir = pathlib.Path(data_dir).with_suffix('')

image_count = len(list(data_dir.glob('*/*.jpg')))
print(f"image_count => {image_count}")

roses = list(data_dir.glob('roses/*'))
PIL.Image.open(str(roses[0]))

# ------------------------------------ Get ready data for train ------------------------------------------------

batch_size = 32
img_height = 180
img_width = 180

# Using 2936 files for training.
train_ds = tf.keras.utils.image_dataset_from_directory(
  data_dir,
  validation_split=0.2,
  subset="training",
  seed=123,
  image_size=(img_height, img_width),
  batch_size=batch_size)

# Using 734 files for validation.
val_ds = tf.keras.utils.image_dataset_from_directory(
  data_dir,
  validation_split=0.2,
  subset="validation",
  seed=123,
  image_size=(img_height, img_width),
  batch_size=batch_size)

class_names = train_ds.class_names
print(f"class_names => {class_names}")

# ----------------------------------- Show some data from dataset -------------------------------------------------

import matplotlib.pyplot as plt

plt.figure(figsize=(10, 10))
for images, labels in train_ds.take(1):
  for i in range(9):
    ax = plt.subplot(3, 3, i + 1)
    plt.imshow(images[i].numpy().astype("uint8"))
    plt.title(class_names[labels[i]])
    plt.axis("off")

for image_batch, labels_batch in train_ds:
  print(f"image_batch shape => {image_batch.shape}")
  print(f"labels_batch shape => {labels_batch.shape}")
  break

plt.show()

# Standardize the data:
normalization_layer = layers.Rescaling(1./255)
normalized_ds = train_ds.map(lambda x, y: (normalization_layer(x), y))
image_batch, labels_batch = next(iter(normalized_ds))
first_image = image_batch[0]
# Notice the pixel values are now in `[0,1]`.
print(f"The pixel values are between:")
print(np.min(first_image), np.max(first_image))

# ----------------------------------- Build Training Model -------------------------------------------------
num_classes = len(class_names)

model = Sequential([
    layers.Conv2D(64, (3, 3), activation='relu', input_shape=(img_height, img_width, 3)),
    layers.MaxPooling2D(),
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dense(num_classes)
])

model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

model.summary()

num_epochs = 20

model_history = model.fit(
  train_ds,
  validation_data=val_ds,
  epochs=num_epochs
)

acc = model_history.history['accuracy']
val_acc = model_history.history['val_accuracy']

loss = model_history.history['loss']
val_loss = model_history.history['val_loss']

epochs_range = range(num_epochs)

plt.figure(figsize=(8, 8))
plt.subplot(1, 2, 1)
plt.plot(epochs_range, acc, label='Training Accuracy')
plt.plot(epochs_range, val_acc, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.title('Training and Validation Accuracy')

plt.subplot(1, 2, 2)
plt.plot(epochs_range, loss, label='Training Loss')
plt.plot(epochs_range, val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.title('Training and Validation Loss')
plt.show()

