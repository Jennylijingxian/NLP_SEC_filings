import os
os.environ["OMP_NUM_THREADS"] = "50"
os.environ["OPENBLAS_NUM_THREADS"] = "50"
os.environ["MKL_NUM_THREADS"] = "50"
os.environ["VECLIB_MAXIMUM_THREADS"] = "50"
os.environ["NUMEXPR_NUM_THREADS"] = "50"
from torch.utils.data import Dataset, DataLoader
from torch.autograd import Variable
import torch
from vocab_extractor import *
import numpy as np
from collections import Counter
from datetime import datetime

saved_input_folder = '/ifs/gsb/usf_interns/test_lr/test_lr5/save_optimizer_version/SavedInputFolder'


class SEC_10K_Dataset(Dataset):

    def __init__(self, files_folder, context_size=4):
        save_glove_vocab_to_pickle()

        words = pickle.load(open(f'{glove_path}/6B.50_words.pkl', 'rb'))
        word2idx = pickle.load(open(f'{glove_path}/6B.50_idx.pkl', 'rb'))
        vectors = pickle.load(open(f'{glove_path}/6B.50_vectors.pkl', 'rb'))

        self.glove = {w: vectors[word2idx[w]] for w in words}

        # set this variable true if want to save input again and false if want to use the already saved input
        self.get_saved_input = True

        if not self.get_saved_input:
            self.word_list_corpus = get_whole_text_as_List(files_folder)
            print("Finished getting the whole corpus",datetime.now())
            self.vocab = list(set(self.word_list_corpus))
            print("Length of vocabulary", len(self.vocab))
            pickle.dump(self.vocab, open(f'{saved_input_folder}/vocab.pkl', 'wb'))
            with open(f'{saved_input_folder}/vocab.txt', 'w') as f:
                f.write(str(self.vocab))

            # Get unigram_table and save it
            Z = 0.001
            unigram_table = []
            word_count = Counter(self.word_list_corpus)
            num_total_words = sum([c for w, c in word_count.items()])
            for vo in self.vocab:
                unigram_table.extend([vo] * int(((word_count[vo] / num_total_words) ** 0.75) / Z))
            self.unigram_table = unigram_table
            pickle.dump(self.unigram_table, open(f'{saved_input_folder}/unigram_table.pkl', 'wb'))
            with open(f'{saved_input_folder}/unigram_table.txt', 'w') as f:
                f.write(str(self.unigram_table))

        else:
            self.vocab = pickle.load(open(f'{saved_input_folder}/vocab.pkl', 'rb'))
            print("Length of vocabulary", len(self.vocab))
            self.unigram_table = pickle.load(open(f'{saved_input_folder}/unigram_table.pkl', 'rb'))

        self.word_to_ix = {word: i for i, word in enumerate(self.vocab)}
        self.ix_to_word = {i:word for i,word in enumerate(self.vocab)}
        self.vocab_size = len(self.vocab)
        self.context_size = context_size

        if not self.get_saved_input:
            X_p = []
            y_p = []
            sindex = 0
            print(len(self.word_list_corpus))
            for i in range(len(self.word_list_corpus) - self.context_size):
                window = self.word_list_corpus[sindex:(sindex + self.context_size + 1)]
                center = window[self.context_size // 2]
                context = window[0:self.context_size // 2] + window[((self.context_size // 2) + 1):]
                for c in context:
                    # Given center word predict context, so x is center, y is context
                    X_p.append(self.prepare_word(center))
                    y_p.append(self.prepare_word(c))
                sindex += 1
            self.train_data = list(zip(X_p, y_p))
            np.save("{saved_input_folder}/train_data.npy",train_data)
            print("Finished processing training data",datetime.now())
        else:
            self.train_data = np.load(f'{saved_input_folder}/train_data.npy')
            print("Finished loading saved training data",datetime.now())

            
    def vocab2idx(self, word):
        return self.word_to_ix[word]
    
    def idx2vocab(self,idx):
        return self.ix_to_word[idx]

    def in_vocab(self, word):
        return word in self.vocab

    def prepare_word(self, word):
        return self.word_to_ix[word]

    def prepare_sequence(self, seq):
        idxs = list(map(lambda w: self.word_to_ix[w], seq))
        return Variable(torch.LongTensor(idxs))

    def __getitem__(self, idx):
        center = torch.from_numpy(np.array(self.train_data[idx][0]))
        context = torch.from_numpy(np.array(self.train_data[idx][1]))
        return center, context

    def __len__(self):
        return len(self.train_data)

    def unigram(self):
        return self.unigram_table

