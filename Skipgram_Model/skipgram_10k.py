import os
os.environ["OMP_NUM_THREADS"] = "50"
os.environ["OPENBLAS_NUM_THREADS"] = "50"
os.environ["MKL_NUM_THREADS"] = "50"
os.environ["VECLIB_MAXIMUM_THREADS"] = "50"
os.environ["NUMEXPR_NUM_THREADS"] = "50"
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from sec_10k_dataset import *


class SkipgramNegSampling(nn.Module):
    def __init__(self, projection_dim, dataset):
        super(SkipgramNegSampling, self).__init__()
        vocab_size = dataset.vocab_size
        weights_matrix = np.zeros((vocab_size, projection_dim))
        words_found = 0
        
        # If word in glove's vocab, use glove embedding, else initialize with random numbers
        for i, word in enumerate(dataset.vocab):
            try:
                weights_matrix[i] = dataset.glove[word]
                words_found += 1
            except KeyError:
                weights_matrix[i] = np.random.normal(scale=0.6, size=(projection_dim,))

        self.embedding_v = nn.Embedding(vocab_size, projection_dim)
        self.embedding_v.load_state_dict({'weight': torch.from_numpy(weights_matrix)})
        self.embedding_u = nn.Embedding(vocab_size, projection_dim)
        self.embedding_u.load_state_dict({'weight': torch.from_numpy(weights_matrix)})
        self.logsigmoid = nn.LogSigmoid()
        print("Finished Initialize model", datetime.now())
        print("Shape of the current embeddings",weights_matrix.shape)

    def forward(self, center_words, target_words, negative_words):
        center_embeds = self.embedding_v(center_words.unsqueeze(1)) # B x 1 x D
        target_embeds = self.embedding_u(target_words.unsqueeze(1)) # B x 1 x D
        neg_embeds = -self.embedding_u(negative_words) # B x K x D
        center_embeds_t = center_embeds.transpose(1, 2) #Bx1
        positive_score = target_embeds.bmm(center_embeds_t).squeeze(2) #Bx1
        negative_score = torch.sum(neg_embeds.bmm(center_embeds_t).squeeze(2), 1) # BxK -> Bx1
        loss = self.logsigmoid(positive_score) + self.logsigmoid(negative_score)
        return -torch.mean(loss)
    
    def get_embeddings(self):
        return self.embedding_v


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
