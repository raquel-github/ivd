import torch
import torch.autograd as autograd
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F

use_cuda = torch.cuda.is_available()
# use_cuda = False

class Decoder(nn.Module):

    def __init__(self, word_embedding_dim, hidden_decoder_dim, visual_features_dim, vocab_size, batch_size=1):
        """
        Parameters
        hidden_encoder_dim      Dimensionaly of the hidden state of the encoder
        hidden_decoder_dim      Dimensionaly of the hidden state of the decoder
        visual_features_dim     Dimensionaly of the visual features
        vocab_size              Size of the vocablurary used for the prediction
        """

        super(Decoder, self).__init__()

        self.word_embedding_dim = word_embedding_dim
        self.hidden_decoder_dim = hidden_decoder_dim
        self.visual_features_dim = visual_features_dim
        self.vocab_size = vocab_size
        self.batch_size = batch_size


        self.decoder_lstm = nn.LSTM(self.word_embedding_dim + self.visual_features_dim, self.hidden_decoder_dim)


        self.hidden2word = nn.Linear(self.hidden_decoder_dim, self.vocab_size)


    def forward(self, visual_features, hidden_encoder=None, decoder_input=None):

        if hidden_encoder:
            # if encoder hidden state is provided, copy into decoder hidden state
            self.hidden_decoder = hidden_encoder

        if decoder_input:
            # if decoder input is provided, copy into previous lstm_out (which will be used as next input)
            self.lstm_out = decoder_input


        # get the input to the LSTM encoder by concatenating word embeddings and visual features
        if use_cuda:
            visual_features = Variable(visual_features.view(1, 1, -1)).cuda()
        else:
            visual_features = Variable(visual_features.view(1, 1, -1))

        decoder_in = torch.cat([self.lstm_out, visual_features], dim=2)

        self.lstm_out, self.hidden_decoder = self.decoder_lstm(decoder_in, self.hidden_decoder)

        # mapping hidden state to word output
        word_space = self.hidden2word(self.lstm_out.view(1,-1))

        # p(w)
        word_scores = F.log_softmax(word_space)

        # print('Decoder Done')
        return word_scores


class DecoderBatch(Decoder):

    def __init__(self, word_embedding_dim, hidden_decoder_dim, visual_features_dim, vocab_size, batch_size):
        Decoder.__init__(self, word_embedding_dim, hidden_decoder_dim, visual_features_dim, vocab_size, batch_size)

        if use_cuda:
            self.decoder_lstm.cuda()
            self.hidden2word.cuda()            

    def forward(self, visual_features, hidden_encoder, decoder_input=None):

        if hidden_encoder:
            # if encoder hidden state is provided, copy into decoder hidden state
            self.hidden_decoder = hidden_encoder

        if decoder_input:
            # if decoder input is provided, copy into previous lstm_out (which will be used as next input)
            self.lstm_out = decoder_input.view(1, self.batch_size, -1)



        visual_features = visual_features.view(1, self.batch_size, -1)


        decoder_in = torch.cat([self.lstm_out, visual_features], dim=2)


        self.lstm_out, self.hidden_decoder = self.decoder_lstm(decoder_in, self.hidden_decoder)

        # mapping hidden state to word output
        word_space = self.hidden2word(self.lstm_out[0])

        # p(w)
        word_scores = F.log_softmax(word_space)

        # print('Decoder Done')
        return word_scores
