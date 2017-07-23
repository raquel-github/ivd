import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


use_cuda = torch.cuda.is_available()
# use_cuda = False


class LSTMQA(nn.Module):
    """docstring for LSTMQA"nn.Module"""
    def __init__(self, vocab_size, word_embedding_dim, hidden_dim, word2index, visual_features_dim, length, batch_size=1):
        super(LSTMQA,self).__init__()

        self.vocab_size = vocab_size
        self.word_embedding_dim = word_embedding_dim
        self.word2index = word2index
        self.hidden_dim = hidden_dim
        self.batch_size = batch_size
        self.length = length

        self.visual_features_dim = visual_features_dim

        # Word embedding Training Model
        if use_cuda:
            self.word_embeddings = nn.Embedding(vocab_size, word_embedding_dim).cuda()
        else:
            self.word_embeddings = nn.Embedding(vocab_size, word_embedding_dim)

        # Initiliaze the hidden state of the LSTM
        self.hidden_state = self.init_hidden()

        self.visual2wordDim = nn.Linear(self.visual_features_dim, self.word_embedding_dim)
        
        self.lstm = nn.LSTM(word_embedding_dim+visual_features_dim, hidden_dim)

        # The linear layer that maps from hidden state space to word space
        self.hidden2word = nn.Linear(hidden_dim, vocab_size)

        if use_cuda:
            self.visual2wordDim.cuda()
            self.lstm.cuda()
            self.hidden2word.cuda()


    def init_hidden(self, train_batch = 1):
        if train_batch:
            if use_cuda:
                return (autograd.Variable(torch.zeros(1, self.batch_size, self.hidden_dim)).cuda(),
                        autograd.Variable(torch.zeros(1, self.batch_size, self.hidden_dim)).cuda())
            else:
                return (autograd.Variable(torch.zeros(1, self.batch_size, self.hidden_dim)),
                        autograd.Variable(torch.zeros(1, self.batch_size, self.hidden_dim)))
        else:
            if use_cuda:
                return (autograd.Variable(torch.zeros(1, self.batch_size, self.hidden_dim), requires_grad=False).cuda(),
                        autograd.Variable(torch.zeros(1, self.batch_size, self.hidden_dim), requires_grad=False).cuda())
            else:
                return (autograd.Variable(torch.zeros(1, self.batch_size, self.hidden_dim), requires_grad=False),
                        autograd.Variable(torch.zeros(1, self.batch_size, self.hidden_dim), requires_grad=False ))


    def forward(self, sentence_batch, visual_features, hidden_state=None):
        """ freivn """
        if use_cuda:
            sentence_batch = Variable(sentence_batch).cuda()
            visual_features = Variable(visual_features).cuda().view(1,-1)
        else:
            sentence_batch = Variable(sentence_batch)
            visual_features = Variable(visual_features).view(1,-1)

        if hidden_state:
            self.hidden_state = hidden_state

        sentence_batch_embedding = self.word_embeddings(sentence_batch).view(len(sentence_batch),self.batch_size,-1)

        if use_cuda:
            visual_featues_words = Variable(torch.zeros(len(sentence_batch), self.batch_size, self.visual_features_dim), requires_grad=False).cuda()
        else:
            visual_featues_words = Variable(torch.zeros(len(sentence_batch), self.batch_size, self.visual_features_dim), requires_grad=False)

        for i in range(len(sentence_batch)):
            visual_featues_words[i] = visual_features

        lstm_in = torch.cat([visual_featues_words, sentence_batch_embedding], dim=2)
        lstm_out, self.hidden_state = self.lstm(lstm_in, self.hidden_state)

        sentence_space = self.hidden2word(lstm_out.view(len(sentence_batch), -1))

        sentence_scores = F.log_softmax(sentence_space)

        return sentence_scores, self.hidden_state      
		
		