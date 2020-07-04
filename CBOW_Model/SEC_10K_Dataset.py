import os
os.environ["OMP_NUM_THREADS"] = "6"
os.environ["OPENBLAS_NUM_THREADS"] = "6"
os.environ["MKL_NUM_THREADS"] = "6"
os.environ["VECLIB_MAXIMUM_THREADS"] = "6"
os.environ["NUMEXPR_NUM_THREADS"] = "6"

from torch.utils.data import Dataset, DataLoader
# from collections import defaultdict
import torch
from vocab_extractor import *
import numpy as np
# import csv

saved_input_folder = 'SavedInputFolder/'

class SEC_10K_Dataset(Dataset):

    def __init__(self, filesFolder, CONTEXT_SIZE = 4):

        save_glove_vocab_to_pickle()

        words = pickle.load(open(f'{glove_path}/6B.50_words.pkl', 'rb'))
        word2idx = pickle.load(open(f'{glove_path}/6B.50_idx.pkl', 'rb'))
        vectors = pickle.load(open(f'{glove_path}/6B.50_vectors.pkl', 'rb'))

        self.glove = {w: vectors[word2idx[w]] for w in words}

        # set this variable true if want to save input again and false if want to use the already saved input
        getSavedInput = False

        if getSavedInput:

            word_list_corpus = get_whole_text_as_List(filesFolder)

            # Create ngrams from test_sentence
            self.context_target = []

            for i in range(len(word_list_corpus) - CONTEXT_SIZE):
                tup1 = [word_list_corpus[j] for j in np.arange(i, (i + CONTEXT_SIZE // 2))]
                tup2 = [word_list_corpus[j] for j in np.arange((i + CONTEXT_SIZE // 2 + 1), (i + CONTEXT_SIZE + 1))]
                tup = tup1 + tup2
                self.context_target.append((tup, word_list_corpus[i + CONTEXT_SIZE // 2]))

            pickle.dump(self.context_target, open(f'{saved_input_folder}/context_target.pkl', 'wb'))

            with open(f'{saved_input_folder}/context_target.txt', 'w') as f:
                f.write(str(self.context_target))


            # Get vocab of test_sentence
            self.vocab = list(set(word_list_corpus))
            print("Length of vocabulary", len(self.vocab))
            pickle.dump(self.vocab, open(f'{saved_input_folder}/vocab.pkl', 'wb'))

            with open(f'{saved_input_folder}/vocab.txt', 'w') as f:
                f.write(str(self.vocab))

        else:
            self.vocab = pickle.load(open(f'{saved_input_folder}/vocab.pkl', 'rb'))
#             file = open('SavedInputFolder/vocab.txt',mode='r')
#             self.vocab = file.read()
#             self.vocab = self.vocab[1:-1].replace(" ", "").replace("'", "").split(",")
        
            self.context_target = pickle.load(open(f'{saved_input_folder}/context_target.pkl', 'rb'))

        self.word_to_ix = {word: i for i, word in enumerate(self.vocab)}
        
        self.idx2vocab = list(self.vocab)
        self.vocab_size = len(self.vocab)

        self.window_size = CONTEXT_SIZE

    def __getitem__(self, idx):

        context = torch.tensor([self.vocab2idx(w) for w in self.context_target[idx][0]])
        target = torch.tensor([self.vocab2idx(self.context_target[idx][1])])
        return context, target

    def __len__(self):
        return len(self.context_target)

    def vocab2idx(self, word):
        # print(word)
#         print("Sakshi:",self.word_to_ix.get(word, -1))
        return self.word_to_ix.get(word, -1)
#         return self.word_to_ix[word]

    def in_vocab(self, word):
        return word in self.vocab
