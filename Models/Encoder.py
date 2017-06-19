import torch
import torch.autograd as autograd
import torch.nn as nn
from torch.autograd import Variable


use_cuda = torch.cuda.is_available()

class Encoder(nn.Module):

    def __init__(self, vocab_size, word_embedding_dim, hidden_encoder_dim, word2index, visual_features_dim):
        """
        Parameters
        vocab_size              Size of the vocablurary
        word_embedding_dim      Size of the word Embeddings
        hidden_encoder_dim      Size of the hidden state of the Encoder
        word2index              Dictionary mapping words in the vocablurary to an index
        visual_features_dim     Dimensionaly of the visual features

        """

        super(Encoder, self).__init__()

        self.vocab_size = vocab_size
        self.word_embedding_dim = word_embedding_dim
        self.hidden_word_embed_layer = int(vocab_size / 2)
        self.visual_features_dim = visual_features_dim
        self.word2index = word2index
        self.hidden_encoder_dim = hidden_encoder_dim

        # Word embedding Training Model
        self.word_embeddings = nn.Embedding(vocab_size, word_embedding_dim)

        # Encoder Model
        self.encoder_lstm = nn.LSTM(word_embedding_dim+visual_features_dim, hidden_encoder_dim)

        # Initiliaze the hidden state of the LSTM
        self.hidden_encoder = self.init_hidden()

    def init_hidden(self):
        if use_cuda:
            return (autograd.Variable(torch.zeros(1, 1, self.hidden_encoder_dim)).cuda(),
                    autograd.Variable(torch.zeros(1, 1, self.hidden_encoder_dim)).cuda())
        else:
            return (autograd.Variable(torch.zeros(1, 1, self.hidden_encoder_dim)),
                    autograd.Variable(torch.zeros(1, 1, self.hidden_encoder_dim)))

    def word2embedd(self, w):
        return self.word_embeddings(Variable(torch.LongTensor([self.word2index[w]])))

    def forward(self, sentence, visual_features):

        # compute the one hot representation of the sentence
        sentence_embedding = Variable(torch.zeros(len(sentence.split()), self.word_embedding_dim))

        for i, w in enumerate(sentence.split()):
            sentence_embedding[i] = self.word2embedd(w)

        # prepare visual features for concatenation
        visual_features_stack = torch.cat([Variable(torch.FloatTensor(visual_features).view(1,-1))] * len(sentence.split()))

        # get the input to the LSTM encoder by concatenating word embeddings and visual features
        encoder_in = torch.cat([sentence_embedding, visual_features_stack], dim=1).view(len(sentence.split()),1,-1)

        # pass word embeddings through encoder LSTM and get output and hidden state
        encoder_out, self.hidden_encoder = self.encoder_lstm(encoder_in, self.hidden_encoder)


        return encoder_out, self.hidden_encoder
