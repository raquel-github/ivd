# Oracle for GuessWhat 
# https://github.com/pytorch/pytorch/issues/619 

import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import json

#TODO: Checken of dit werkt
from Preprocessing.DataReader import DataReader

from torchtext.vocab import load_word_vectors

import numpy
import h5py

def train():
    # Load data
    dr = DataReader()

    #Settings LSTM
    hidden_dim = 128 #Dit staat bij encoder van guesser iig

    visual_len = 4096
    #object_len = 
    category_len = 
    spatial_len = 4
    embedding_dim = 128
    vocab_size = 4200
    
    #Settings MLP
    d_in = visual_len + spatial_len + category_len + hidden_dim #+ object_len
    d_hin = (d_in+d_out)/2 #mean is nu wel erg groot?
    d_hidden = (d_hin+d_out)/2
    d_hout = (d_hidden+d_out)/2
    d_out = 3

    #Instance of Oracle om LSTM en MLP te runnen?
    model = Oracle(embedding_dim,hidden_dim, d_in, d_hin, d_hidden, d_hout, d_out)
    loss = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters())

    #Get Game/Question and run model
    gameids = dr.get_game_ids()
    for gid in gameids:
        image = dr.get_image_features(gid)
        #crop = dr.get_crop_features(gid) #TODO
        obj = get_target_object(gid)
        quas = dr.get_questions(gid)
        answers = dr.get_answers(gid)
        for question in quas:
            outputs = model.forward(question, spatial_info, object_class, crop, image)
            answer = answers(question) #TODO: answers en questions pairen?
            cost = loss(outputs,answer)

            # Backpropogate Errors ||TODO: also applies to LSTM?
            optimizer.zero_grad() 
            cost.backward()
            optimizer.step()

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
        sentence_embedding = Variable(torch.zeros(len(sentence.split()), self.word_embedding_dim))

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
        #TODO: Dimenties checken
        mlp_in = torch.cat([encoder_out, visual_features, crop_features, spatial_info, object_class])

        # MLP pass
        mlp_out = self.mlp(mlp_in) 
        #TODO: turn mlp_out into "Yes", "No", "N/A" softmax dus.
        return mlp_out 


if __name__ == '__main__':
    train()


