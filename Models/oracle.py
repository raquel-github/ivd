import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from torch.autograd import Variable

import json
import numpy
import h5py

use_cuda = torch.cuda.is_available()

class Oracle(nn.Module):

    # hidden_dim: Dimensionality of output of LSTM block.
    # embedding_dim: (inputsize LSTM)
    # d_in: De lengte van de totale inputvector (MLP)
    # d_hin/d_hidden/d_hout: dimenties van hidden layer: 
    # --- helft van de dimensies die het verbind, recursively, voor gradual overgang.
    # d_out: 3 (Yes,No,N/A)
    def __init__(self, vocab_size, embedding_dim, categories_length, object_embedding_dim, hidden_dim, d_in, d_hin, d_hidden, d_hout, d_out, word2index):
        # Dit weet ik allemaal niet zo goed meer: is dit nodig?
        super(Oracle, self).__init__()
        self.hidden_dim = hidden_dim
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.object_embedding_dim = object_embedding_dim
        self.word2index = word2index
        self.object_embedding_model = nn.Embedding(categories_length, object_embedding_dim)
        
        # Word Embeddings
        self.word_embeddings = nn.Embedding(vocab_size, embedding_dim)

        # LSTM model that encodes Question
        self.lstm = nn.LSTM(embedding_dim, hidden_dim) 

        # MLP model that classifies to an Answer
        self.mlp = nn.Sequential(
            nn.Linear(int(d_in), int(d_hin)),
            nn.ReLU(), 
            nn.Linear(int(d_hin), int(d_hidden)), 
            nn.ReLU(), 
            nn.Linear(int(d_hidden), int(d_hout)),
            nn.ReLU(),
            nn.Linear(int(d_hout), int(d_out))
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
        # Compute representation of the sentence
        sentence_embedding = Variable(torch.zeros(len(question.split()), self.embedding_dim))
        for i, w in enumerate(question.split()):
            sentence_embedding[i] = self.word2embedd(w)

        # print(sentence_embedding)
        encoder_in = sentence_embedding.view(len(question.split()), 1, -1)
        
        # LSTM pass
        hidden = self.init_hidden() 
        encoder_out = self.lstm(encoder_in, hidden)

        # print(self.object_embedding_model)

        #Answer question
        # object_class = self.object_embedding_model(autograd.Variable(torch.LongTensor(int(object_class))))
        mlp_in = Variable(torch.cat([image, crop, spatial_info, torch.FloatTensor(int(object_class)), encoder_out]))

        # MLP pass
        mlp_out = self.mlp(mlp_in) 
        return mlp_out 





