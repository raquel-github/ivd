# Oracle for GuessWhat 
# https://github.com/pytorch/pytorch/issues/619 

import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import json

import numpy
import h5py

class Oracle(nn.Module()):

    # hidden_dim: Dimensionality of output of LSTM block.
    # embedding_dim: 100? Nu dus vocabulary size? (LSTM)
    # d_in: De lengte van de totale inputvector (MLP)
    # d_hin/d_hidden/d_hout: dimenties van hidden layer: 
    # --- helft van de dimensies die het verbind, recursively, voor gradual overgang.
    # d_out: 3 (Yes,No,N/A)
    def __init__(self, embedding_dim,hidden_dim, d_in, d_hin, d_hidden, d_hout, d_out)
        # Dit weet ik allemaal niet zo goed meer: is dit nodig?
        super(Oracle, self).__init__()
        self.hidden_dim = hidden_dim

        # Word Embeddings
        self.word_embeddings = nn.Embedding(vocab_size, embedding_dim)

        # LSTM model that encodes Question
        self.lstm = nn.LSTM(embedding_dim,hidden_dim) 
        self.hidden = self.init_hidden 

        # MLP model that classifies to an Answer
        self.mlp = nn.Sequential(
            nn.Linear(d_in, d_hin),
            nn.ReLU(), 
            nn.Linear(d_hin, d_hidden), 
            nn.ReLU(), 
            nn.Linear(d_hidden, d_hout),
            nn.ReLU(),
            nn.Linear(d_hout, d_out)
        )

    def word2embedd(self, w):
        return self.word_embeddings(Variable(torch.LongTensor([self.word2index[w]])))

    def init_hidden(self):
        if use_cuda:
            return (autograd.Variable(torch.zeros(1, 1, self.hidden_dim)).cuda(),
                    autograd.Variable(torch.zeros(1, 1, self.hidden_dim)).cuda())
        else:
            return (autograd.Variable(torch.zeros(1, 1, self.hidden_dim)),
                    autograd.Variable(torch.zeros(1, 1, self.hidden_dim)))

    def forward(self, question, spatial_info, object_class, crop, image, training=True):
        # Compute the one hot representation of the sentence
        sentence_embedding = Variable(torch.zeros(len(question.split()), self.word_embedding_dim))

        for i, w in enumerate(sentence.split()):
            if w == '-SOS-':
                sentence_embedding[i] = self.sos
            else:
                sentence_embedding[i] = self.word2embedd(w)

        # Get tensor with word embeddings
        encoder_in = self.word_embed_model(sent_onehot)
        
        # LSTM pass
        encoder_out, self.hidden = self.lstm(encoder_in, self.hidden)

        #Answer question
        mlp_in = torch.cat([encoder_out, image, crop, spatial_info, object_class])

        # MLP pass
        mlp_out = self.mlp(mlp_in) 
        #TODO: turn mlp_out into "Yes", "No", "N/A" softmax dus.
        return mlp_out 





