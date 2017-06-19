import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy

class QGen(nn.Module):

    def __init__(self, embedding_dim, hidden_dim, vocab_size, target_size):

        """
        Parameters
        embedding_dim       Size of the word embedding input
        hidden_dim          Size of the LSTM hidden state
        vocab_size          Size of the vocablurary
        target_size         Sie of the p(w) output distribution
        """



        super(QGen, self).__init__()
        self.hidden_dim = hidden_dim

        # Word Embeddings Training Layer
        self.embedding_training = nn.Sequential(
            nn.Linear(vocab_size, embedding_hidden_size),
            nn.ReLU(),
            nn.Linear(embedding_hidden_size, word_embedding_dim)
        )

        # Encoder
        self.encoder = nn.LSTM(word_embedding_dim, hidden_dim)


        # The LSTM takes word embeddings as inputs, and outputs hidden states
        # with dimensionality hidden_dim.
        self.lstm = nn.LSTM(embedding_dim+feature_dim, hidden_dim)

        # The linear layer that maps from hidden state space to word space
        self.hidden2word = nn.Linear(hidden_dim, target_size)
        self.hidden = self.init_hidden()

    def init_hidden(self):
        if use_cuda:
            return (autograd.Variable(torch.zeros(1, 1, self.hidden_dim)).cuda(),
                    autograd.Variable(torch.zeros(1, 1, self.hidden_dim)).cuda())
        else:
            return (autograd.Variable(torch.zeros(1, 1, self.hidden_dim)),
                    autograd.Variable(torch.zeros(1, 1, self.hidden_dim)))

    def forward(self, sentence, hidden=None):

        if hidden == None:
            hidden = self.hidden

        # Word embedding of sentence
        if use_cuda:
            embeds = autograd.Variable(torch.FloatTensor(len(sentence.split()), embedding_dim)).cuda()
        else:
            embeds = autograd.Variable(torch.FloatTensor(len(sentence.split()), embedding_dim))


        for i, w in enumerate(sentence.split()):
            embeds[i] = get_wv(w)

        
        # LSTM
        if use_cuda:
            visual_features = autograd.Variable(torch.randn(len(sentence.split()), 1, feature_dim)).cuda()
        else:
            visual_features = autograd.Variable(torch.randn(len(sentence.split()), 1, feature_dim))


        word_embeddings = embeds.view(len(sentence.split()), 1, -1)

        lstm_in = torch.cat([word_embeddings, visual_features], dim=2)


        lstm_out, self.hidden = self.lstm(lstm_in, hidden)


        # mapping hidden state to word output
        word_space = self.hidden2word(lstm_out.view(len(sentence.split()), -1))


        # p(w)
        word_scores = F.log_softmax(word_space)

        return word_scores, self.hidden
