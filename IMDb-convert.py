# Usage: python IMDb-convert.py
# Description: This script converts the IMDb dataset on keras into a more readable format.
# 
# The dataset and word index are already downloaded using functions from tf.keras.datasets.imdb (Data Sources),
# and saved under the 'data' directory as 'imdb.npz' and 'imdb_word_index.json'.
# The program loads the dataset, decodes the reviews from their integer representation to words.
# It saves the converted data to the 'data' directory as 'imdb_converted.npz' with the same format as 'imdb.npz'.
# 
# Data Sources: https://ai.stanford.edu/%7Eamaas/data/sentiment/
#               https://www.tensorflow.org/api_docs/python/tf/keras/datasets/imdb

import numpy as np
import json

def load_dataset(file_path):
    data = np.load(file_path, allow_pickle=True)
    x_train = data['x_train']
    y_train = data['y_train']
    x_test = data['x_test']
    y_test = data['y_test']
    return x_train, y_train, x_test, y_test

def load_word_index(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        word_index = json.load(file)
    return word_index

def construct_inverted_word_index(word_index):
    inverted_word_index = dict((value, key) for (key, value) in word_index.items())
    return inverted_word_index

def decode(encoded_seq, inverted_word_index):
    decoded_seq = " ".join(inverted_word_index[i] for i in encoded_seq)
    decoded_seq = " ".join(decoded_seq.split())
    return decoded_seq

def decode_dataset(dataset, inverted_word_index):
    decoded_dataset = []
    for encoded_seq in dataset:
        decoded_seq = decode(encoded_seq, inverted_word_index)
        decoded_dataset.append(decoded_seq)
    return decoded_dataset

def save_processed_data(file_path, x_train, y_train, x_test, y_test):
    np.savez(file_path, x_train=x_train, y_train=y_train, x_test=x_test, y_test=y_test)

(x_train, y_train, x_test, y_test) = load_dataset('data/imdb.npz')
word_index = load_word_index('data/imdb_word_index.json')
inverted_word_index = construct_inverted_word_index(word_index)

x_train = decode_dataset(x_train, inverted_word_index)
x_test = decode_dataset(x_test, inverted_word_index)

x_train = np.array(x_train)
x_test = np.array(x_test)
y_train = np.array(y_train)
y_test = np.array(y_test)
save_processed_data('data/imdb_converted.npz', x_train, y_train, x_test, y_test)
