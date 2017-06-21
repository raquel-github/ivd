import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F

use_cuda = torch.cuda.is_available()

class DecoderAttn(nn.Module):

    def __init__(self, hidden_encoder_dim, hidden_decoder_dim, vocab_size, word_embedding_dim, max_length):
        """
        Parameters
        hidden_encoder_dim      Dimensionaly of the hidden state of the encoder
        """

        super(DecoderAttn, self).__init__()

        self.hidden_encoder_dim = hidden_encoder_dim
        self.hidden_decoder_dim = hidden_decoder_dim
        self.word_embedding_dim = word_embedding_dim
        self.vocab_size = vocab_size
        self.max_length = max_length

        self.attn = nn.Linear(self.hidden_decoder_dim * 2, self.max_length)
        self.attn_combine = nn.Linear(self.hidden_decoder_dim * 2, self.hidden_decoder_dim)

        self.decoder_lstm = nn.LSTM(hidden_encoder_dim, hidden_decoder_dim)

        self.hidden2word = nn.Linear(hidden_decoder_dim, vocab_size)


    def forward(self, encoder_outputs, hidden_encoder=None, decoder_input=None):


        if hidden_encoder:
            # if encoder hidden state is provided, copy into decoder hidden state
            self.hidden_decoder = hidden_encoder

        if decoder_input:
            # if decoder input is provided, copy into previous lstm_out (which will be used as next input)
            self.lstm_out = decoder_input

        # get attention weights
        attn_weights = F.log_softmax(self.attn(torch.cat([self.hidden_decoder[0][0], self.lstm_out[0]], dim=1)))

        # multiply with encoder outputs
        attn_applied = torch.bmm(attn_weights.unsqueeze(0), encoder_outputs.unsqueeze(0))

        # concat attention with input and reduce to input size with one linear transformation
        self.lstm_out = torch.cat((self.lstm_out[0], attn_applied[0]), 1)
        self.lstm_out = self.attn_combine(self.lstm_out).unsqueeze(0)

        # pass through decoder
        self.lstm_out, self.hidden_decoder = self.decoder_lstm(self.lstm_out, self.hidden_decoder)


        # mapping hidden state to word output
        word_space = self.hidden2word(self.lstm_out.view(1,-1))

        # p(w)
        word_scores = F.log_softmax(word_space)

        return word_scores
