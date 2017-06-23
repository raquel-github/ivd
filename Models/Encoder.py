import torch
import torch.autograd as autograd
import torch.nn as nn
from torch.autograd import Variable


use_cuda = torch.cuda.is_available()
# use_cuda = False

class Encoder(nn.Module):

    def __init__(self, vocab_size, word_embedding_dim, hidden_encoder_dim, word2index, batch_size=1):
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
        self.word2index = word2index
        self.hidden_encoder_dim = hidden_encoder_dim
        self.batch_size = batch_size

        # Word embedding Training Model
        if use_cuda:
            self.word_embeddings = nn.Embedding(vocab_size, word_embedding_dim).cuda()
        else:
            self.word_embeddings = nn.Embedding(vocab_size, word_embedding_dim)

        # Encoder Model
        self.encoder_lstm = nn.LSTM(word_embedding_dim, hidden_encoder_dim)

        # Initiliaze the hidden state of the LSTM
        self.hidden_encoder = self.init_hidden()

        if use_cuda:
            self.sos = Variable(torch.randn(self.word_embedding_dim,1)).view(1,1,-1).cuda()
        else:
            self.sos = Variable(torch.randn(self.word_embedding_dim,1)).view(1,1,-1)

    def init_hidden(self):
        if use_cuda:
            return (autograd.Variable(torch.zeros(1, self.batch_size, self.hidden_encoder_dim)).cuda(),
                    autograd.Variable(torch.zeros(1, self.batch_size, self.hidden_encoder_dim)).cuda())
        else:
            return (autograd.Variable(torch.zeros(1, self.batch_size, self.hidden_encoder_dim)),
                    autograd.Variable(torch.zeros(1, self.batch_size, self.hidden_encoder_dim)))

    def word2embedd(self, w):
        if use_cuda:
            return self.word_embeddings(Variable(torch.LongTensor([self.word2index[w]])).cuda())
        else:
            return self.word_embeddings(Variable(torch.LongTensor([self.word2index[w]])))

    def forward(self, sentence):

        # compute the one hot representation of the sentence
        if use_cuda:
            sentence_embedding = Variable(torch.zeros(len(sentence.split()), self.word_embedding_dim)).cuda()
        else:
            sentence_embedding = Variable(torch.zeros(len(sentence.split()), self.word_embedding_dim))

        for i, w in enumerate(sentence.split()):
            sentence_embedding[i] = self.word2embedd(w)


        sentence_embedding = sentence_embedding.view(len(sentence.split()),1,-1)

        # pass word embeddings through encoder LSTM and get output and hidden state
        encoder_out, self.hidden_encoder = self.encoder_lstm(sentence_embedding, self.hidden_encoder)

        return encoder_out, self.hidden_encoder


class EncoderBatch(Encoder):
    """docstring for E"""
    def __init__(self, vocab_size, word_embedding_dim, hidden_encoder_dim, word2index, batch_size):
        # super(EncoderBatch, self).__init__()
        Encoder.__init__(self,vocab_size, word_embedding_dim, hidden_encoder_dim, word2index, batch_size)


    def forward(self, sentence_batch):
        """ freivn """
        sentence_batch_embedding = self.word_embeddings(sentence_batch)
        print(sentence_batch_embedding.size())
        print(self.hidden_encoder[0].size())
        #packed = torch.nn.utils.rnn.pack_padded_sequence(sentence_batch_embedding, input_lengths)
        encoder_out, self.hidden_encoder = self.encoder_lstm(sentence_batch_embedding, self.hidden_encoder)

        return encoder_out, self.hidden_encoder
