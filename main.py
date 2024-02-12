import random
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import sequential 
from tensorflow.keras.layers import LSTM, Dense, Activation
from tensorflow.keras.optimizers import RMSprop

filepath = tf.keras.utils.get_file('shakespea.txt','https://storage.googleapis.com/download.tensorflow.org/data/shakespeare.txt')

text = open(filepath, 'rb').read().decode(encoding= 'utf-8' ).lower()

#now we want to connect this into a neuralnetwork 
#convert it into a unique numerical representation

text = text [3000000:800000]

characters = sorted(set(text))

char_to_index = dict((c,i) for i, c in enumerate(characters))

