import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torchvision import transforms, utils

import pickle

use_cuda = torch.cuda.is_available()
# use_cuda = False

#Contains the word vectors from GLOVE with embedding dimension of 300 specifically containg words from vocabulary of Oracle training data stored as numpy array
wv_file = 'vocab_wv.pickle'  

class Oracle(nn.Module):
    """docstring for Oracle"""
    def __init__(self, word_embedding_dim, obj_cat_embedding_dim, hidden_lstm_dim, vocab_size, obj_cat_size):
        super(Oracle, self).__init__()

        self.vocab_size = vocab_size
        self.word_embedding_dim = word_embedding_dim
        self.hidden_lstm_dim = hidden_lstm_dim
        self.obj_cat_size = obj_cat_size
        self.obj_cat_embedding_dim = obj_cat_embedding_dim

        # Word embedding Training Model
        self.word_embeddings = nn.Embedding(self.vocab_size, self.word_embedding_dim)

        with open(wv_file, 'rb') as file:
            self.word_embeddings.weight.data.copy_(torch.from_numpy(pickle.load(file)))

        self.obj_cat_embedding = nn.Embedding(self.obj_cat_size, self.obj_cat_embedding_dim)

        # LSTM Model
        self.lstm = nn.LSTM(self.word_embedding_dim, self.hidden_lstm_dim)

        # Initiliaze the hidden state of the LSTM
        # self.hidden_lstm = self.init_hidden(64)

        self.mlp1 = nn.Linear((4096*2)+self.word_embedding_dim+self.obj_cat_embedding_dim+8, 256)
        self.mlp2 = nn.Linear(256,256)
        self.mlp3 = nn.Linear(256,3)
        # self.mlp4 = nn.Linear(500,3)

    def init_hidden(self, actual_batch_size, split = 'train'):
        if split == 'train':
            if use_cuda:
                return (Variable(torch.zeros(1, actual_batch_size, self.hidden_lstm_dim)).cuda(),
                        Variable(torch.zeros(1, actual_batch_size, self.hidden_lstm_dim)).cuda())
            else:
                return (Variable(torch.zeros(1, actual_batch_size, self.hidden_lstm_dim)),
                        Variable(torch.zeros(1, actual_batch_size, self.hidden_lstm_dim)))
        else:
            if use_cuda:
                return (Variable(torch.zeros(1, actual_batch_size, self.hidden_lstm_dim), requires_grad=False).cuda(),
                            Variable(torch.zeros(1, actual_batch_size, self.hidden_lstm_dim), requires_grad=False ).cuda())
            else:
                return (Variable(torch.zeros(1, actual_batch_size, self.hidden_lstm_dim), requires_grad=False),
                            Variable(torch.zeros(1, actual_batch_size, self.hidden_lstm_dim), requires_grad=False ))


    def forward(self, split, question_batch, obj_cat_batch, spatial_batch, crop_features, image_features, actual_batch_size):
        if use_cuda:
            question_batch = Variable(question_batch).cuda()
            obj_cat_batch = Variable(obj_cat_batch).cuda()
            spatial_batch = Variable(spatial_batch).cuda()
            crop_features = Variable(crop_features, requires_grad=False).cuda()
            image_features = Variable(image_features, requires_grad=False).cuda()
        else:
            question_batch = Variable(question_batch)
            obj_cat_batch = Variable(obj_cat_batch)
            spatial_batch = Variable(spatial_batch)
            crop_features = Variable(crop_features, requires_grad=False)
            image_features = Variable(image_features, requires_grad=False)
    
        question_batch_embedding = self.word_embeddings(question_batch)
        question_batch_embedding = torch.mean(question_batch_embedding, 1)
        # print(question_batch_embedding)
        obj_cat_batch_embeddding  = self.obj_cat_embedding(obj_cat_batch)

        # self.hidden_lstm = self.init_hidden(actual_batch_size, split)

        # print(question_batch_embedding.size())
        # _, self.hidden_lstm = self.lstm(question_batch_embedding.view(46, actual_batch_size, self.word_embedding_dim), self.hidden_lstm) # 46 == Max length of the question. 

        mlp_in = torch.cat([ image_features, crop_features, spatial_batch, obj_cat_batch_embeddding, question_batch_embedding], 1)

        mlp_out = F.relu(self.mlp1(mlp_in))
        mlp_out = F.relu(self.mlp2(mlp_out))
        mlp_out = F.relu(self.mlp3(mlp_out))
        # mlp_out = F.relu(self.mlp4(mlp_out))

        return F.log_softmax(mlp_out)
