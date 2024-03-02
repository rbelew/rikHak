
import os

os.environ["KERAS_BACKEND"] = "torch" # "tensorflow"

import keras
import tensorflow as tf
import numpy as np
from keras import layers

import string
import re

import torch

import socket
HOST = socket.gethostname()

if HOST=='OSX_M1_HOST':
    DEVICE = torch.device("mps")
    # DEVICE = torch.device("cpu")
elif HOST=='LINUX_HOST':
    # DEVICE = torch.device("cuda")
    DEVICE = torch.device("cpu")

# Model constants.
max_features = 20000
embedding_dim = 128
sequence_length = 500

def custom_standardization(input_data):
    lowercase = tf.strings.lower(input_data)
    stripped_html = tf.strings.regex_replace(lowercase, "<br />", " ")
    return tf.strings.regex_replace(
        stripped_html, f"[{re.escape(string.punctuation)}]", ""
    )

vectorize_layer = keras.layers.TextVectorization(
    standardize=custom_standardization,
    max_tokens=max_features,
    output_mode="int",
    output_sequence_length=sequence_length,
)


def vectorize_text(text, label):
    text = tf.expand_dims(text, -1)
    return vectorize_layer(text), label


def main():
    
    kerasNLPDir = '.../keras-nlp/'
    
    batch_size = 32
    raw_train_ds = keras.utils.text_dataset_from_directory(
        kerasNLPDir+"aclImdb/train",
        batch_size=batch_size,
        validation_split=0.2,
        subset="training",
        seed=1337,
    )
    raw_val_ds = keras.utils.text_dataset_from_directory(
        kerasNLPDir+"aclImdb/train",
        batch_size=batch_size,
        validation_split=0.2,
        subset="validation",
        seed=1337,
    )
    raw_test_ds = keras.utils.text_dataset_from_directory(
        kerasNLPDir+"aclImdb/test", batch_size=batch_size
    )
    
    # Inspect first review
    # Format is (review text tensor, label tensor)
    print(raw_train_ds.unbatch().take(1).get_single_element())
    
    print(f"Number of batches in raw_train_ds: {raw_train_ds.cardinality()}")
    print(f"Number of batches in raw_val_ds: {raw_val_ds.cardinality()}")
    print(f"Number of batches in raw_test_ds: {raw_test_ds.cardinality()}")
    
    for text_batch, label_batch in raw_train_ds.take(1):
        for i in range(5):
            print(text_batch.numpy()[i])
            print(label_batch.numpy()[i])


    text_ds = raw_train_ds.map(lambda x, y: x)
    vectorize_layer.adapt(text_ds)
    
    # Vectorize the data.
    train_ds = raw_train_ds.map(vectorize_text)
    val_ds = raw_val_ds.map(vectorize_text)
    test_ds = raw_test_ds.map(vectorize_text)
    
    # Do async prefetching / buffering of the data for best performance on GPU.
    train_ds = train_ds.cache().prefetch(buffer_size=10)
    val_ds = val_ds.cache().prefetch(buffer_size=10)
    test_ds = test_ds.cache().prefetch(buffer_size=10)
    
    # A integer input for vocab indices.
    inputs = keras.Input(shape=(None,), dtype="int64")
    
    # Next, we add a layer to map those vocab indices into a space of dimensionality
    # 'embedding_dim'.
    x = layers.Embedding(max_features, embedding_dim,name='embed')(inputs)
    x = layers.Dropout(0.5)(x)
    
    # Conv1D + global max pooling
    x = layers.Conv1D(128, 7, padding="valid", activation="relu", strides=3,name='conv1')(x)
    x = layers.Conv1D(128, 7, padding="valid", activation="relu", strides=3,name='conv2')(x)
    x = layers.GlobalMaxPooling1D()(x)
    
    # We add a vanilla hidden layer:
    x = layers.Dense(128, activation="relu",name='dense')(x)
    x = layers.Dropout(0.5,name='drop')(x)
    
    # We project onto a single unit output layer, and squash it with a sigmoid:
    predictions = layers.Dense(1, activation="sigmoid", name="predictions")(x)
    
    model = keras.Model(inputs, predictions)

    model.to(DEVICE)
    
    # if DEVICE !=torch.device("cpu"):
    #     for p in model.parameters():
    #         print (p, p.is_mps if DEVICE ==torch.device("mps") else p.is_cuda)
    
    # Compile the model with binary crossentropy loss and an adam optimizer.
    model.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])
    
    print(f'DEVICE={DEVICE} type={type(DEVICE)}')
    epochs = 3
    # Fit the model using the train and test datasets.
    model.fit(train_ds, validation_data=val_ds, epochs=epochs)
    
  
    model.evaluate(test_ds)

if __name__ == '__main__':
    main()
