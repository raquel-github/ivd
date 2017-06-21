import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F

use_cuda = torch.cuda.is_available()

class Decoder(nn.Module):

    def __init__(self, word_embedding_dim, hidden_decoder_dim, vocab_size):
        """
        Parameters
        hidden_encoder_dim      Dimensionaly of the hidden state of the encoder
        """

        super(Decoder, self).__init__()

        self.word_embedding_dim = word_embedding_dim
        self.hidden_decoder_dim = hidden_decoder_dim
        self.vocab_size = vocab_size

        self.decoder_lstm = nn.LSTM(self.word_embedding_dim, self.hidden_decoder_dim)
        

        self.hidden2word = nn.Linear(self.hidden_decoder_dim, self.vocab_size)


    def forward(self, hidden_encoder=None, decoder_input=None):

        if hidden_encoder:
            # if encoder hidden state is provided, copy into decoder hidden state
            self.hidden_decoder = hidden_encoder

        if decoder_input:
            # if decoder input is provided, copy into previous lstm_out (which will be used as next input)
            self.lstm_out = decoder_input

        self.lstm_out, self.hidden_decoder = self.decoder_lstm(self.lstm_out, self.hidden_decoder)

        # mapping hidden state to word output
        word_space = self.hidden2word(self.lstm_out.view(1,-1))

        # p(w)
        word_scores = F.log_softmax(word_space)

        return word_scores
