# tst_240306: wayne: torch CPU vs MTL, TF on CPU only

import socket
HOST = socket.gethostname()

import numpy as np
import string
import re
import sys

import os
os.environ["KERAS_BACKEND"] = "torch" # "torch" # "tensorflow"

import keras
from keras import layers

import tensorflow as tf

import torch

import torchtext
import keras_nlp
import tensorflow_text

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

if os.environ["KERAS_BACKEND"] == 'torch':
    # DEVICE = torch.device("mps")
    # print(f'device torch mps fixed: {DEVICE == torch.device("mps")}')
    DEVICE = torch.device("cpu")
    print(f'device torch cpu fixed: {DEVICE == torch.device("cpu")}')

elif os.environ["KERAS_BACKEND"] == 'tensorflow':
    # DEVICE = tf.config.list_physical_devices("GPU")[0]
    # print(f'device fixed: {DEVICE == tf.config.list_physical_devices("GPU")[0]}')
    DEVICE = tf.config.list_physical_devices("CPU")[0]
    print(f'device TF CPU fixed: {DEVICE == tf.config.list_physical_devices("CPU")[0]}')
    

if os.environ["KERAS_BACKEND"] == 'torch' and DEVICE == torch.device("mps"):
    print(f"{HOST} torch MPS {str(DEVICE)} avail={torch.backends.mps.is_available()} built={torch.backends.mps.is_built()}")

# elif os.environ["KERAS_BACKEND"] == 'tensorflow' and DEVICE == tf.config.list_physical_devices('GPU')[0]:
#     tf.debugging.set_log_device_placement(True)
#     print(f"{HOST} TF MPS NGPUs Available: {len(tf.config.list_physical_devices('GPU'))}")

elif os.environ["KERAS_BACKEND"] == 'tensorflow' and DEVICE != tf.config.list_physical_devices("CPU")[0]:
    print('keras-tensorflow doesnt work on MTL!?')
    sys.exit(-1)


print(f'''\ttorch={torch.__version__}
            torchtext={torchtext.__version__}
            tensorflow={tf.__version__}
            tensorflow_text={tensorflow_text.__version__}
            keras={keras.__version__}
            keras_nlp={keras_nlp.__version__}
    ''')
    

max_features = 20000
embedding_dim = 128
sequence_length = 500
  
with torch.device("cpu"): 
# with tf.device("CPU"):
 
    kerasNLPDir = '/Users/rik/data/corpora/keras-nlp/'
    
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
         kerasNLPDir+"aclImdb/test", 
         batch_size=batch_size,
     )    

    text_ds = raw_train_ds.map(lambda x, y: x)
    vectorize_layer.adapt(text_ds)

    # Vectorize the data.
    train_ds = raw_train_ds.map(vectorize_text)
    val_ds = raw_val_ds.map(vectorize_text)
    test_ds = raw_test_ds.map(vectorize_text) 
   
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
    # Compile the model with binary crossentropy loss and an adam optimizer.
    model.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])
  
    epochs = 2
    # Fit the model using the train and test datasets.
    model.fit(train_ds, validation_data=val_ds, epochs=epochs)
    
    model.evaluate(test_ds)
    
    import pdb; pdb.set_trace()

print('done')
            
