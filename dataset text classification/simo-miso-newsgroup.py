import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
from sklearn.datasets import fetch_20newsgroups

epoch_num=10
# Download and prepare the 20 Newsgroups dataset
print("20 Newsgroups dataset")
newsgroups = fetch_20newsgroups(subset='all', remove=('headers', 'footers', 'quotes'))
texts = newsgroups.data
labels = newsgroups.target

# Tokenize and pad sequences
max_words = 10000  # Adjust as needed
max_len = 200  # Adjust as needed
tokenizer = Tokenizer(num_words=max_words)
tokenizer.fit_on_texts(texts)
sequences = tokenizer.texts_to_sequences(texts)
text_data = pad_sequences(sequences, maxlen=max_len)

# Split the data into training and testing sets
text_train, text_test, label_train, label_test = train_test_split(text_data, labels, test_size=0.2, random_state=42)

# ----------------------------------- Build Multiple Input Single Output (MISO) Model -------------------------------------------------

# Define the model
text_input = layers.Input(shape=(max_len,), name='text_input')
embedding_layer = layers.Embedding(10000, 16)(text_input)
flatten_layer = layers.Flatten()(embedding_layer)
dense_layer = layers.Dense(32, activation='relu')(flatten_layer)
output_layer = layers.Dense(1, activation='sigmoid', name='output')(dense_layer)

miso_model = models.Model(inputs=text_input, outputs=output_layer)

miso_model.compile(optimizer='adam',
                   loss='binary_crossentropy',
                   metrics=['accuracy'])

miso_model.summary()

# Train the MISO model
miso_model_history = miso_model.fit(
    x=text_train,
    y=label_train,
    validation_data=(text_test, label_test),
    epochs=epoch_num
)

# ----------------------------------- Build Single Input Multiple Output (SIMO) Model -------------------------------------------------

# Define the SIMO model
output1 = layers.Dense(20, activation='softmax', name='output1')(dense_layer)
output2 = layers.Dense(1, activation='sigmoid', name='output2')(dense_layer)

simo_model = models.Model(inputs=text_input, outputs=[output1, output2])

simo_model.compile(optimizer='adam',
                   loss={'output1': 'sparse_categorical_crossentropy', 'output2': 'binary_crossentropy'},
                   metrics={'output1': 'accuracy', 'output2': 'accuracy'})

simo_model.summary()

# Train the SIMO model
simo_model_history = simo_model.fit(
    x=text_train,
    y={'output1': label_train, 'output2': label_train},
    validation_data=(text_test, {'output1': label_test, 'output2': label_test}),
    epochs=epoch_num
)

# Plot training and validation accuracy and loss for both models

# MISO Model
plt.figure(figsize=(8, 8))
plt.subplot(2, 2, 1)
plt.plot(miso_model_history.history['accuracy'], label='Training Accuracy')
plt.plot(miso_model_history.history['val_accuracy'], label='Validation Accuracy')
plt.legend(loc='lower right')
plt.title('MISO Model - Training and Validation Accuracy')

plt.subplot(2, 2, 2)
plt.plot(miso_model_history.history['loss'], label='Training Loss')
plt.plot(miso_model_history.history['val_loss'], label='Validation Loss')
plt.legend(loc='upper right')
plt.title('MISO Model - Training and Validation Loss')

# SIMO Model
plt.subplot(2, 2, 3)
plt.plot(simo_model_history.history['output1_accuracy'], label='Output1 Training Accuracy')
plt.plot(simo_model_history.history['output2_accuracy'], label='Output2 Training Accuracy')
plt.plot(simo_model_history.history['val_output1_accuracy'], label='Output1 Validation Accuracy')
plt.plot(simo_model_history.history['val_output2_accuracy'], label='Output2 Validation Accuracy')
plt.legend(loc='lower right')
plt.title('SIMO Model - Training and Validation Accuracy')

plt.subplot(2, 2, 4)
plt.plot(simo_model_history.history['loss'], label='Training Loss')
plt.plot(simo_model_history.history['output1_loss'], label='Output1 Training Loss')
plt.plot(simo_model_history.history['output2_loss'], label='Output2 Training Loss')
plt.plot(simo_model_history.history['val_loss'], label='Validation Loss')
plt.plot(simo_model_history.history['val_output1_loss'], label='Output1 Validation Loss')
plt.plot(simo_model_history.history['val_output2_loss'], label='Output2 Validation Loss')
plt.legend(loc='upper right')
plt.title('SIMO Model - Training and Validation Loss')

plt.tight_layout()
plt.show()
