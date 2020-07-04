import os
import torch
import numpy as np
import urllib.request
from nltk.tokenize import RegexpTokenizer
import pickle
import re
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
import string
import nltk
# nltk.download('punkt')
from nltk.stem.porter import *
import codecs

torch.manual_seed(1)

CONTEXT_SIZE = 4
glove_path = 'glove.6B'

def save_glove_vocab_to_pickle():
    words = []
    vectors = []
    idx = 0
    word2idx = {}
    # vectors = bcolz.carray(np.zeros(1), rootdir=f'{glove_path}/6B.50.dat', mode='w')

    with open(f'{glove_path}/glove.6B.50d.txt', 'rb') as f:
        for l in f:
            line = l.decode().split()
            word = line[0]
            words.append(word)
            word2idx[word] = idx
            idx += 1
            vect = np.array(line[1:]).astype(np.float)
            vectors.append(vect)

    # vectors = bcolz.carray(vectors[1:].reshape((400000, 50)), rootdir=f'{glove_path}/6B.50.dat', mode='w')
    # vectors.flush()
    pickle.dump(vectors, open(f'{glove_path}/6B.50_vectors.pkl', 'wb'))
    pickle.dump(words, open(f'{glove_path}/6B.50_words.pkl', 'wb'))
    pickle.dump(word2idx, open(f'{glove_path}/6B.50_idx.pkl', 'wb'))


def read_data(file_path):
    tokenizer = RegexpTokenizer(r'\w+')
    # data = open(file_path, 'r')
    # data = urllib.request.urlopen(file_path)
    # text = data.read()
    text = codecs.open(file_path, 'r', encoding='utf-8', errors='ignore').read()
    text = text.lower()
    text = re.sub('[' + string.punctuation + '0-9\\r\\t\\n\®]', ' ', text)
    tokens = nltk.word_tokenize(text)
    tokens = [w for w in tokens if len(w) > 2]  # ignore a, an, to, at, be, ...
    goodWords = [w for w in tokens if w not in ENGLISH_STOP_WORDS]
    return (goodWords)


def get_whole_text_as_List(filesFolder):

    fnames = os.listdir(filesFolder)
    complete_test_sentence = []
    for file in fnames:
        print("tokenizing file:", file)
        test_sentence = read_data(filesFolder + file)
        complete_test_sentence = complete_test_sentence+test_sentence
    return complete_test_sentence
