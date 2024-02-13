from keras import layers, models
from keras.layers import Input, Embedding, Flatten, Dense, concatenate
from keras.datasets import imdb
from keras.preprocessing.sequence import pad_sequences
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.text import Tokenizer
from sklearn.model_selection import train_test_split
from sklearn.datasets import fetch_20newsgroups
# ----------------------------------- Load and Preprocess IMDb dataset -------------------------------------------------
epoch_num = 10
print("20 Newsgroups dataset")
max_words = 10000
max_len = 200

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
text_train, text_test, train_labels, test_labels = train_test_split(text_data, labels, test_size=0.2, random_state=42)


# (train_data, train_labels), (test_data, test_labels) = imdb.load_data(num_words=max_words)
train_data = pad_sequences(text_train, maxlen=max_len)
test_data = pad_sequences(text_test, maxlen=max_len)

# Split the data into two parts
half_len = len(train_data) // 2
train_data_text, train_data_extra = train_data[:half_len], train_data[half_len:]
test_data_text, test_data_extra = test_data[:half_len], test_data[half_len:]

# ----------------------------------- Build Multiple Input Single Output (MISO) Model -------------------------------------------------

# Define the model
text_input = Input(shape=(max_len,), name='text_input')
extra_input = Input(shape=(max_len,), name='extra_input')

# Embedding layer for text input
embedding_layer = Embedding(max_words, 16)(text_input)
flatten_layer = Flatten()(embedding_layer)

# Concatenate the flattened embedding layer with the extra input
merged_input = concatenate([flatten_layer, extra_input])

dense_layer = Dense(32, activation='relu')(merged_input)
output_layer = Dense(1, activation='sigmoid', name='output')(dense_layer)

# Build the MISO model
miso_model = models.Model(inputs=[text_input, extra_input], outputs=output_layer)

miso_model.compile(optimizer='adam',
                   loss='binary_crossentropy',
                   metrics=['accuracy'])

miso_model.summary()

# Train the MISO model
miso_model_history = miso_model.fit(
    x=[train_data_text, train_data_extra],
    y=train_labels[:half_len],  # Use labels corresponding to text_input only
    validation_data=([test_data_text, test_data_extra], test_labels[:half_len]),  # Use labels corresponding to text_input only
    epochs=epoch_num
)
