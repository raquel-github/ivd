import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F

class Decoder(nn.Module):

    def __init__(self, hidden_encoder_dim, hidden_decoder_dim):
        """
        Parameters
        hidden_encoder_dim      Dimensionaly of the hidden state of the encoder
        """

        super(Decoder, self).__init__()

        self.hidden_encoder_dim = encoder_dim
        self.hidden_decoder_dim = decoder_dim

        self.decoder_lstm = nn.LSTM(hidden_encoder_dim, hidden_decoder_dim)

        self.hidden_decoder = init_hidden()

        self.hidden2word = nn.Linear(hidden_decoder_dim, vocab_size)


    def init_hidden(self):
        if use_cuda:
            return (autograd.Variable(torch.zeros(1, 1, self.hidden_decoder_dim)).cuda(),
                    autograd.Variable(torch.zeros(1, 1, self.hidden_decoder_dim)).cuda())
        else:
            return (autograd.Variable(torch.zeros(1, 1, self.hidden_decoder_dim)),
                    autograd.Variable(torch.zeros(1, 1, self.hidden_decoder_dim)))


    def forward(self, hidden_encoder):

        lstm_out, self.hidden_decoder = self.decoder_lstm(hidden_encoder, self.hidden_decoder)

        # mapping hidden state to word output
        word_space = self.hidden2word(lstm_out.view(len(sentence.split()), -1))

        # p(w)
        word_scores = F.log_softmax(word_space)


        return word_scores
