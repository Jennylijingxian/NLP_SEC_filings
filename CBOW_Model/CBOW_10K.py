import os
os.environ["OMP_NUM_THREADS"] = "6" 
os.environ["OPENBLAS_NUM_THREADS"] = "6" 
os.environ["MKL_NUM_THREADS"] = "6" 
os.environ["VECLIB_MAXIMUM_THREADS"] = "6" 
os.environ["NUMEXPR_NUM_THREADS"] = "6" 

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from SEC_10K_Dataset import *


class CBOW_10K(nn.Module):

    def __init__(self, embedding_dim, dataset):
        super(CBOW_10K, self).__init__()
        emb_dim = embedding_dim
        matrix_len = dataset.vocab_size
        weights_matrix = np.zeros((matrix_len, embedding_dim))
        words_found = 0

        for i, word in enumerate(dataset.vocab):
            try:
                weights_matrix[i] = dataset.glove[word]
                words_found += 1
            except KeyError:
                weights_matrix[i] = np.random.normal(scale=0.6, size=(emb_dim,))

        self.embeddings = nn.Embedding(dataset.vocab_size, embedding_dim)
        self.embeddings.load_state_dict({'weight': torch.from_numpy(weights_matrix)})

        self.linear = nn.Linear(embedding_dim, dataset.vocab_size)

        self.activation = nn.LogSoftmax(dim=1)

    def forward(self, inputs):
        out = torch.sum(self.embeddings(inputs), dim=1)
        out = self.linear(out)
        return self.activation(out)

    def get_embeddings(self):
        return self.embeddings


# From https://rguigoures.github.io/word2vec_pytorch/

class EarlyStopping():
    def __init__(self, patience=5, min_percent_gain=0.1):
        self.patience = patience
        self.loss_list = []
        self.min_percent_gain = min_percent_gain / 100.

    def update_loss(self, loss):
        self.loss_list.append(loss)
        if len(self.loss_list) > self.patience:
            del self.loss_list[0]

    def stop_training(self):
        if len(self.loss_list) == 1:
            return False
        gain = (max(self.loss_list) - min(self.loss_list)) / max(self.loss_list)
        print("Loss gain: {}%".format(round(100 * gain, 2)))
        if gain < self.min_percent_gain:
            return True
        else:
            return False
