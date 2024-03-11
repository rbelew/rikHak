'''tst_240311: using SentencePieceTokenizer with LegalBench dataset
'''

import socket
HOST = socket.gethostname()

from collections import OrderedDict

import numpy as np
import string
import re
import sys

import os
## NB: critical to specify KERAS_BACKEND *BEFORE* importing keras!
os.environ["KERAS_BACKEND"] = "tensorflow"

import keras
from keras import layers

import tensorflow as tf
import tensorflow_text

# from tensorflow_models import nlp

import torch
import torchtext
from torchtext.data.functional import load_sp_model
from torchtext.vocab import vocab
from torchtext.transforms import VocabTransform
from torchtext.models import RobertaClassificationHead

import keras_nlp

from keras_nlp.tokenizers import SentencePieceTokenizer

import datasets # HuggingFace
from datasets import Dataset

SPtokenizer = None
   
def main():
    
    print(f'''\ttorch={torch.__version__}
                torchtext={torchtext.__version__}
                tensorflow={tf.__version__}
                tensorflow_text={tensorflow_text.__version__}
                keras={keras.__version__}
                keras_nlp={keras_nlp.__version__}
        ''')
        
    LHTaskDir = '/Users/rik/data/git/legalbench-main/tasks/learned_hands_torts/'
    SPmodelFile = '/Users/rik/data/ai4law/caseLaw/ill_text_20210921/IllCL.model'
    SPvocabFile = '/Users/rik/data/ai4law/caseLaw/ill_text_20210921/IllCL.vocab'

    # UsingLegalBench.ipynb
    batch_size = 16
    max_seq_len = 100
    bos_idx = 0
    eos_idx = 99
    
    embed_size = 100

    LH_dataset_HF = datasets.load_dataset("nguha/legalbench", 'learned_hands_torts')
    
    # other things I've tried(:
    # train_tensor =  LH_dataset_HF['train'][:]
    # dsTF = LH_dataset_HF.with_format("tf")
    
    trainTF = LH_dataset_HF['train'].with_format("tf")
             
    trainDS = LH_dataset_HF['train'].to_tf_dataset(
            columns=["text"],
            label_cols=["answer"],
            batch_size=batch_size,
            shuffle=False,
            )
           
    global SPtokenizer
    SPtokenizer = SentencePieceTokenizer(proto=SPmodelFile,sequence_length=32)
    SPvocabDict = {SPtokenizer.id_to_token(i): i for i in range(SPtokenizer.vocabulary_size())}
    SPvocabList = SPtokenizer.get_vocabulary()
 
    print(f"trainTF shape={trainTF.shape} answer shape={trainTF['answer'].shape}")
    print(f"all answers={trainTF['answer']}")
    
    LH_Torts_Train0 = LH_dataset_HF['train']['text'][0]
    # '''My roommate and I were feeling unwell in our basement apartment for a long time. We discovered a drier was exhausting directly into our unit. We asked the landlord to fix it, but he did, and ever since then it has gotten way worse. There's a chemical smell in the air and staying in the apt more than ~15 mins causes extreme fatigue, loss of focus, sinuses closing up, and chest tightness (Whatever it is isn't triggering the CO2 or natural gas alarm)    Landlord agreed to terminate our lease and let us keep our stuff in there unpaid for the first few weeks but now he wants us to set a move out date and pack up. But I feel really bad when I stay in here. The last time I went in to get my stuff I fell asleep suddenly and woke up 2 hours later with a nosebleed and difficulty breathing.    Landlord refuses to hire an indoor air quality inspector and says he plans on sealing off the whole basement and no longer renting it. I called the housing inspector but he said he only inspects whole houses, not just the basement... so I would need to go through my landlord... what do I do?'''
    input_text = LH_Torts_Train0
        
    tokens = SPtokenizer.tokenize(input_text)
    input_ids = SPtokenizer.detokenize(tokens)
    echo1 = tf.Variable(input_ids).numpy()
    print('echo1: ',echo1)
    
    inputs = keras.Input(shape=(), dtype="string")
    
    x = SPtokenizer(inputs)
    
    x = layers.Embedding(input_dim=SPtokenizer.vocabulary_size(), output_dim=embed_size,name="embed")(x)
 
    predictions = layers.Dense(1, activation="sigmoid", name="predictions")(x)
    
    model = keras.Model(inputs,predictions)

    model.summary()
    model.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])
    
    model.fit(trainDS, epochs=3)
    
    model.evaluate(test_ds)
    
           
if __name__ == '__main__':
    main()
