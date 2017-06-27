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

    def init_hidden(self, train_batch = 1):
        if train_batch:
            if use_cuda:
                return (autograd.Variable(torch.zeros(1, self.batch_size, self.hidden_encoder_dim)).cuda(),
                        autograd.Variable(torch.zeros(1, self.batch_size, self.hidden_encoder_dim)).cuda())
            else:
                return (autograd.Variable(torch.zeros(1, self.batch_size, self.hidden_encoder_dim)),
                        autograd.Variable(torch.zeros(1, self.batch_size, self.hidden_encoder_dim)))
        else:
            if use_cuda:
                return (autograd.Variable(torch.zeros(1, self.batch_size, self.hidden_encoder_dim), volatile=True).cuda(),
                        autograd.Variable(torch.zeros(1, self.batch_size, self.hidden_encoder_dim), volatile=True).cuda())
            else:
                return (autograd.Variable(torch.zeros(1, self.batch_size, self.hidden_encoder_dim), volatile=True),
                        autograd.Variable(torch.zeros(1, self.batch_size, self.hidden_encoder_dim), volatile=True))

    def word2embedd(self, w):
        if use_cuda:
            return self.word_embeddings(Variable(torch.LongTensor([self.word2index[w]])).cuda())
        else:
            return self.word_embeddings(Variable(torch.LongTensor([self.word2index[w]])))

    def get_sos_embedding(self, use_cuda):
        if use_cuda:
            return self.word_embeddings(Variable(torch.LongTensor([int(self.word2index['-SOS-'])]*self.batch_size),requires_grad=False).cuda())
        else:
            return self.word_embeddings(Variable(torch.LongTensor([int(self.word2index['-SOS-'])]*self.batch_size),requires_grad=False))

    def forward(self, sentence, visual_features):

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
    def __init__(self, vocab_size, word_embedding_dim, hidden_encoder_dim, word2index, visual_features_dim, length, batch_size):
        # super(EncoderBatch, self).__init__()
        Encoder.__init__(self,vocab_size, word_embedding_dim, hidden_encoder_dim, word2index, batch_size)

        self.visual_features_dim = visual_features_dim

        self.visual2wordDim = nn.Linear(self.visual_features_dim, self.word_embedding_dim)

        self.encoder_lstm = nn.LSTM(word_embedding_dim*2, hidden_encoder_dim)

        self.length = length

    def forward(self, sentence_batch, visual_features_batch, hidden_state_encoder=None):
        """ freivn """
        if use_cuda:
            sentence_batch = Variable(sentence_batch).cuda()
        else:
            sentence_batch = Variable(sentence_batch)

        if hidden_state_encoder:
            self.hidden_encoder = hidden_state_encoder

        sentence_batch_embedding = self.word_embeddings(sentence_batch)

        if use_cuda:
            visual_featues_batch_words = Variable(torch.zeros(self.length+1, self.batch_size, self.word_embedding_dim), requires_grad=False).cuda()
        else:
            visual_featues_batch_words = Variable(torch.zeros(self.length+1, self.batch_size, self.word_embedding_dim), requires_grad=False)

        for i in range(self.length+1):
            visual_featues_batch_words[i] = self.visual2wordDim(visual_features_batch)

        encoder_in = torch.cat([sentence_batch_embedding, visual_featues_batch_words], dim=2)

        encoder_out, self.hidden_encoder = self.encoder_lstm(encoder_in, self.hidden_encoder)

        return encoder_out, self.hidden_encoder
