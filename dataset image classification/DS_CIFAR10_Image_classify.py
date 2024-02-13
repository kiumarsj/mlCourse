import numpy as np
import psutil
import matplotlib.pyplot as plt
# TensorFlow and keras
import tensorflow as tf
from keras import layers, datasets
from keras.models import Sequential
print(f"Tensorflow version => {tf.__version__}")

import memory_profiler

@profile
def main():

  # ------------------------------------ Get ready dataset ------------------------------------------------

  # The CIFAR10 dataset contains 60,000 color images in 10 classes
  print("CIFAR10 dataset")
  (train_images, train_labels), (test_images, test_labels) = datasets.cifar10.load_data()

  # Normalize pixel values to be between 0 and 1
  train_images, test_images = train_images / 255.0, test_images / 255.0

  class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer',
                'dog', 'frog', 'horse', 'ship', 'truck']

  # ----------------------------------- Show some data from dataset -------------------------------------------------

  plt.figure(figsize=(10,10))
  for i in range(25):
      plt.subplot(5,5,i+1)
      plt.xticks([])
      plt.yticks([])
      plt.grid(False)
      plt.imshow(train_images[i])
      # The CIFAR labels happen to be arrays, 
      # which is why you need the extra index
      plt.xlabel(class_names[train_labels[i][0]])
  # plt.show()

  img_height, img_width, d = np.shape(train_images[0])
  print(f"Shape of image: {np.shape(train_images[0])}")

  # ----------------------------------- Build Training Model -------------------------------------------------

  num_classes = len(class_names)

  model = Sequential([
      layers.Conv2D(32, (3, 3), activation='relu', input_shape=(img_height, img_width, 3)),
      layers.MaxPooling2D(),
      layers.Flatten(),
      layers.Dense(64, activation='relu'),
      layers.Dense(num_classes)
  ])

  model.compile(optimizer='adam',
                loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                metrics=['accuracy'])

  model.summary()

  num_epochs = 30

  model_history = model.fit(
    x = train_images, 
    y = train_labels,
    validation_data=(test_images, test_labels),
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
  # plt.show()

main()
print(f"CPU Usage: {psutil.cpu_percent()}%")