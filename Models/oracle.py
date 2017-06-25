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
    def __init__(self, vocab_size, embedding_dim, categories_length, object_embedding_dim, hidden_dim, d_in, d_hin, d_hidden, d_hout, d_out, word2index, batch_size=1):
        # Dit weet ik allemaal niet zo goed meer: is dit nodig?
        super(Oracle, self).__init__()
        self.hidden_dim = hidden_dim
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.object_embedding_dim = object_embedding_dim
        self.word2index = word2index
        self.batch_size = batch_size

        # Embeddings
        if use_cuda:
            self.object_embedding_model = nn.Embedding(categories_length, object_embedding_dim).cuda()
            self.word_embeddings = nn.Embedding(vocab_size, embedding_dim).cuda()
        else:
            self.object_embedding_model = nn.Embedding(categories_length, object_embedding_dim)
            self.word_embeddings = nn.Embedding(vocab_size, embedding_dim)


        # LSTM model that encodes Question
        self.lstm = nn.LSTM(embedding_dim, hidden_dim) 

        self.hidden = self.init_hidden()

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
        if use_cuda:
            return self.word_embeddings(Variable(torch.LongTensor([self.word2index[w]])).cuda())
        else:
            return self.word_embeddings(Variable(torch.LongTensor([self.word2index[w]])))

    def obj2embedd(self,obj):
        if use_cuda:
            return self.object_embedding_model(Variable(torch.LongTensor([int(obj)])).cuda())
        else:
            return self.object_embedding_model(Variable(torch.LongTensor([int(obj)])))

    def init_hidden(self):
        if use_cuda:
            return (autograd.Variable(torch.zeros(1, 1, self.hidden_dim)).cuda(),
                    autograd.Variable(torch.zeros(1, 1, self.hidden_dim)).cuda())
        else:
            return (autograd.Variable(torch.zeros(1, 1, self.hidden_dim)),
                    autograd.Variable(torch.zeros(1, 1, self.hidden_dim)))

    def forward(self, question, spatial, object_class, crop, image):
        # Compute representation of the sentence
        if use_cuda:
            sentence_embedding = Variable(torch.zeros(len(question.split()), self.embedding_dim)).cuda()
        else:
            sentence_embedding = Variable(torch.zeros(len(question.split()), self.embedding_dim))
        
        for i, w in enumerate(question.split()):
            sentence_embedding[i] = self.word2embedd(w)

        # print(sentence_embedding)
        encoder_in = sentence_embedding.view(len(question.split()), 1, -1)
        
        # LSTM pass
        _ , hidden  = self.lstm(encoder_in, self.hidden)

        # Format data
        object_class = self.obj2embedd(object_class)

        if use_cuda:
            image = image.view(1, -1).cuda()
            crop = crop.view(1, -1).cuda()
            spatial = spatial.view(1,-1).cuda()
        else:
            image = image.view(1, -1)
            crop = crop.view(1, -1)
            spatial = spatial.view(1,-1)

        hidden_lstm = hidden[0].view(1,-1)

        #Get answer

        if use_cuda:
            mlp_in = Variable(torch.cat([image, crop, spatial.data, object_class.data, hidden_lstm.data],dim=1)).cuda()
        else:
            mlp_in = Variable(torch.cat([image, crop, spatial.data, object_class.data, hidden_lstm.data],dim=1))

        # MLP pass
        mlp_out = self.mlp(mlp_in) 
        return mlp_out 

class OracleBatch(Oracle):
    def __init__(self, vocab_size, embedding_dim, categories_length, object_embedding_dim, hidden_dim, d_in, d_hin, d_hidden, d_hout, d_out, word2index, batch_size):
        Oracle.__init__(self, vocab_size, embedding_dim, categories_length, object_embedding_dim, hidden_dim, d_in, d_hin, d_hidden, d_hout, d_out, word2index, batch_size)

    def forward(self, question, spatial, object_class, crop, image, num):

        # Save the result for all qa pairs in the batch
        out = torch.Tensor(num, 3)
        
        # Loop over all QA pairs
        for i in range(num):

            if use_cuda:
                sentence_embedding = Variable(torch.zeros(len(question[i].split()), self.embedding_dim)).cuda()
            else:
                sentence_embedding = Variable(torch.zeros(len(question[i].split()), self.embedding_dim))
            
            for j, w in enumerate(question[i].split()):
                sentence_embedding[j] = self.word2embedd(w)

            # print(sentence_embedding)
            encoder_in = sentence_embedding.view(len(question[i].split()), 1, -1)
            
            # LSTM pass
            _ , hidden  = self.lstm(encoder_in, self.hidden)

            # Format data
            object_class_emb = self.obj2embedd(object_class[i])

            if use_cuda:
                image_emb = image[i].view(1, -1).cuda()
                crop_emb = crop[i].view(1, -1).cuda()
                spatial_emb = spatial[i].view(1,-1).cuda()
            else:
                image_emb = image[i].view(1, -1)
                crop_emb = crop[i].view(1, -1)
                spatial_emb = spatial[i].view(1,-1)

            hidden_lstm = hidden[0].view(1,-1)

            #Get answer

            if use_cuda:
                mlp_in = Variable(torch.cat([image_emb, crop_emb, spatial_emb.data, object_class_emb.data, hidden_lstm.data],dim=1)).cuda()
            else:
                mlp_in = Variable(torch.cat([image_emb, crop_emb, spatial_emb.data, object_class_emb.data, hidden_lstm.data],dim=1))

            # MLP pass
            out[i] = self.mlp(mlp_in).data

        # Return the results
        return Variable(out) 
