import os
os.environ["OMP_NUM_THREADS"] = "50"
os.environ["OPENBLAS_NUM_THREADS"] = "50"
os.environ["MKL_NUM_THREADS"] = "50"
os.environ["VECLIB_MAXIMUM_THREADS"] = "50"
os.environ["NUMEXPR_NUM_THREADS"] = "50"
import os
import torch
import numpy as np
from nltk.tokenize import RegexpTokenizer
import pickle
import re
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
import string
import nltk
from nltk.stem.porter import *
import codecs
from nltk.stem import WordNetLemmatizer

torch.manual_seed(1)

CONTEXT_SIZE = 4
glove_path = '/ifs/gsb/usf_interns/test_lr/glove.6B'
printable = set(string.printable)

def save_glove_vocab_to_pickle():
    """Save glove embedding to pickle files.
    """
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

    pickle.dump(vectors, open(f'{glove_path}/6B.50_vectors.pkl', 'wb'))
    pickle.dump(words, open(f'{glove_path}/6B.50_words.pkl', 'wb'))
    pickle.dump(word2idx, open(f'{glove_path}/6B.50_idx.pkl', 'wb'))

def read_data(file_path):
    """Read files, remove words with length less than 2 and special characters.
       Input: file path of the documents.
       Output: cleaned list of words in sequence.
    """
    text = codecs.open(file_path, 'r', encoding='utf-8', errors='ignore').read()
    valid_characters = string.printable
    text = ''.join(i for i in text if i in valid_characters)
    text = text.lower()
    text = re.sub('[' + string.punctuation + '0-9\\r\\t\\n\®]', ' ', text)
    tokens = nltk.word_tokenize(text)
    tokens = [w for w in tokens if len(w) > 2]  # ignore a, an, to, at, be, ...
#     lemmatizer = WordNetLemmatizer()
#     tokens = [lemmatizer.lemmatize(w) for w in tokens]
    goodWords = [w for w in tokens if w not in ENGLISH_STOP_WORDS]
    # goodWords2 = remove_non_ascii(goodWords)
    return (goodWords)


def get_whole_text_as_List(filesFolder):
    """Piece together all the text from multiple documents. 
       Input: file path of the documents.
       Output: full text in a list format
    """
    fnames = os.listdir(filesFolder)
    complete_test_sentence = []
    for file in fnames:
        print("tokenizing file:", file)
        test_sentence = read_data(filesFolder + file)
        complete_test_sentence = complete_test_sentence+test_sentence
    return complete_test_sentence


