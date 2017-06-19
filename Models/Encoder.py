import torch
import torch.autograd as autograd
import torch.nn as nn

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
        self.hidden_encoder = hidden_encoder
        self.visual_features_dim = visual_features_dim

        # Word embedding Training Model
        self.word_embed_model = nn.Sequential(
            nn.Linear(vocab_size, hidden_word_embed_layer),
            nn.ReLU(),
            nn.Linear(hidden_word_embed_layer, word_embedding_dim),
            nn.Tanh()
        )

        # Encoder Model
        self.encoder_lstm = nn.LSTM(word_embedding_dim+visual_features_dim, hidden_encoder_dim)

        # Initiliaze the hidden state of the LSTM
        self.hidden_encoder = init_hidden()

    def init_hidden(self):
        if use_cuda:
            return (autograd.Variable(torch.zeros(1, 1, self.hidden_encoder_dim)).cuda(),
                    autograd.Variable(torch.zeros(1, 1, self.hidden_encoder_dim)).cuda())
        else:
            return (autograd.Variable(torch.zeros(1, 1, self.hidden_encoder_dim)),
                    autograd.Variable(torch.zeros(1, 1, self.hidden_encoder_dim)))

    def word2onehot(self, w):
        onehot = torch.zeros(self.vocab_size)
        onehot[word2index[w]] = 1
        return onehot

    def forward(self, sentence, visual_features):

        # compute the one hot representation of the sentence
        sent_onehot = torch.zeros(len(sentence.split()), self.vocab_size)
        for i, w in enumerate(sentence.split()):
            sent_onehot[i] = word2index(w)

        # get word embedding
        word_embed = self.word_embed_model(sent_onehot)

        # prepare visual features for concatenation
        visual_features_stack = torch.cat([visual_features.view(1, -1)] * len(sentence.split()))

        # get the input to the LSTM encoder by concatenating word embeddings and visual features
        encoder_in = torch.cat([word_embed, visual_features])

        # pass word embeddings through encoder LSTM and get output and hidden state
        encoder_out, self.hidden_encoder = self.encoder_lstm(encoder_in, self.hidden_encoder)


        return encoder_out, self.hidden_encoder
