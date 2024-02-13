import numpy as np
import matplotlib.pyplot as plt
from keras import layers, models
from keras.datasets import imdb
from keras.layers import Input, Embedding, Flatten, Dense, concatenate
from keras.preprocessing.sequence import pad_sequences

# ----------------------------------- Load and Preprocess IMDb dataset -------------------------------------------------
epoch_num = 10
print("IMDb dataset")
max_words = 10000
max_len = 200

(train_data, train_labels), (test_data, test_labels) = imdb.load_data(num_words=max_words)
train_data = pad_sequences(train_data, maxlen=max_len)
test_data = pad_sequences(test_data, maxlen=max_len)

# ----------------------------------- Build Multiple Input Single Output (MISO) Model -------------------------------------------------

# Define the model
text_input = layers.Input(shape=(max_len,), name='text_input')
embedding_layer = layers.Embedding(max_words, 16)(text_input)
flatten_layer = layers.Flatten()(embedding_layer)
dense_layer = layers.Dense(32, activation='relu')(flatten_layer)
output_layer = layers.Dense(1, activation='sigmoid', name='output')(dense_layer)

rating_out = layers.Dense(1, activation="sigmoid", name="rating")(dense_layer) 
sentiment_out = layers.Dense(1, activation="sigmoid", name="sentiment")(dense_layer)

# miso_model = models.Model(inputs=text_input, outputs=output_layer)
# miso_model.compile(optimizer='adam',
#                    loss='binary_crossentropy',
#                    metrics=['accuracy'])

# miso_model.summary()

# # Train the MISO model
# miso_model_history = miso_model.fit(
#     x=train_data,
#     y=train_labels,
#     validation_data=(test_data, test_labels),
#     epochs=epoch_num
# )

# Split the data into two parts
half_len = len(train_data) // 2
train_data_text, train_data_extra = train_data[:half_len], train_data[half_len:]
test_data_text, test_data_extra = test_data[:half_len], test_data[half_len:]

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



# miso_model = models.Model(text_input, [rating_out, sentiment_out])

# miso_model.compile(
#   optimizer="adam",
#   loss={"rating":"binary_crossentropy", "sentiment":"binary_crossentropy"}, 
#   metrics={"rating":"accuracy", "sentiment":"accuracy"}  
# )


# # Split labels into rating and sentiment
# train_ratings = train_labels[:len(train_labels)//2] 
# train_sentiments = train_labels[len(train_labels)//2:]
# test_ratings = test_labels[:len(test_labels)//2]
# test_sentiments = test_labels[len(test_labels)//2:]

# # Train MISO Model
# miso_model_history = miso_model.fit(
#   train_data, 
#   {"rating":train_ratings, "sentiment":train_sentiments},
#   epochs=10,
#   validation_data=(test_data, {"rating":test_ratings, "sentiment":test_sentiments})
# )

# ----------------------------------- Build Single Input Multiple Output (SIMO) Model -------------------------------------------------

# Define the SIMO model
output1 = layers.Dense(1, activation='sigmoid', name='output1')(dense_layer)
output2 = layers.Dense(1, activation='sigmoid', name='output2')(dense_layer)

simo_model = models.Model(inputs=text_input, outputs=[output1, output2])

simo_model.compile(optimizer='adam',
                   loss={'output1': 'binary_crossentropy', 'output2': 'binary_crossentropy'},
                   metrics={'output1': 'accuracy', 'output2': 'accuracy'})

simo_model.summary()

# Train the SIMO model
simo_model_history = simo_model.fit(
    x=train_data,
    y={'output1': train_labels, 'output2': train_labels},
    validation_data=(test_data, {'output1': test_labels, 'output2': test_labels}),
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
