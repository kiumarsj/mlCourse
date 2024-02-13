import sqlite3
import psutil
import memory_profiler

@profile
def main():
    DATABASE_FILE = "DB_of_images.db"
    conn = sqlite3.connect(DATABASE_FILE)
    cursor = conn.cursor()
    cursor.execute('SELECT * FROM image')
    dataset = cursor.fetchall()
    conn.close()

    # ------------------------------------------------------------------------------------
    import numpy as np
    import matplotlib.pyplot as plt
    import cv2
    from io import BytesIO
    # TensorFlow and keras
    import tensorflow as tf
    from keras import layers
    from keras.models import Sequential
    import keras.utils as IMG
    print(tf.__version__)
    from sklearn.model_selection import train_test_split


    samples = []
    labels = []
    for img_obj in dataset:
        labels.append(int(img_obj[3]))
        img_bytes = img_obj[2]
        img = IMG.load_img(BytesIO(img_bytes), target_size=(299, 299))
        # img.show()
        img_normalized = cv2.normalize(np.array(img), None, 0, 1.0, cv2.NORM_MINMAX, dtype=cv2.CV_32F)
        # plt.imshow(img)
        samples.append(img_normalized)

    class_names = ['Human', 'Animal', 'Food']

    # Checking true normalizing images (0.0 and 1.0)
    print(np.min(samples[0]), np.max(samples[0]))

    # show some of samples
    plt.figure(figsize=(10,10))
    for i in range(25):
        plt.subplot(5,5,i+1)
        plt.xticks([])
        plt.yticks([])
        plt.grid(False)
        plt.imshow(samples[i])
        plt.xlabel(class_names[labels[i]])
    # plt.show()

    train_images, test_images, train_labels, test_labels = train_test_split(samples, labels, test_size=0.1, random_state=42)

    train_images = np.array(train_images)
    test_images = np.array(test_images)
    train_labels = np.array(train_labels)
    test_labels = np.array(test_labels) 

    print("\ntrain_images => ", train_images.shape)
    print("train_labels => ", train_labels.shape)
    print("test_images => ", test_images.shape)
    print("test_labels => ", test_labels.shape)

    # Build the model
    num_classes = len(class_names)

    model = Sequential([
        layers.Flatten(input_shape=(299, 299, 3)),
        layers.Dense(128, activation='relu'),
        layers.Dense(3)
    ])

    model.compile(optimizer='adam',
                loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                metrics=['accuracy'])

    model.summary()

    num_epochs = 30

    model_history = model.fit(
    x=train_images,
    y=train_labels,
    validation_data=(test_images, test_labels),
    epochs=num_epochs
    )

    test_loss, test_acc = model.evaluate(test_images,  test_labels, verbose=2)
    # print('\nTest accuracy:', test_acc)

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