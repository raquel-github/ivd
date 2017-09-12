import torch
import torch.nn as nn
from torch.autograd import Variable

class Oracle(nn.Module):

    def __init__(self, hidden_dim, word_emb_dim, cat_emb_dim, vocab_size, cat_size):

        super(Oracle, self).__init__()

        self.hidden_dim = hidden_dim

        self.word_embeddings = nn.Embedding(vocab_size, word_emb_dim)
        self.cat_embeddings = nn.Embedding(cat_size, cat_emb_dim)

        self.lstm = nn.LSTM(word_emb_dim, hidden_dim)

        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim+8+cat_emb_dim, 512),
            nn.ReLU(),
            nn.Linear(512,3),
            nn.LogSoftmax()
        )

    def init_hidden(self, batch_size):
        return (Variable(torch.zeros(1, batch_size, self.hidden_dim)),
                Variable(torch.zeros(1, batch_size, self.hidden_dim))
               )

    def forward(self, questions, categories, spatials):

        batch_size = questions.size(0)
        sequence_length = 10

        hidden_state = self.init_hidden(batch_size)


        print(self.word_embeddings(Variable(questions)).view(sequence_length, batch_size, -1).size())
        print(hidden_state[0].size())

        _, (hidden, _) = self.lstm(self.word_embeddings(Variable(questions)).view(sequence_length, batch_size, -1), hidden_state)

        print(hidden.size())
        print(spatials.size())
        print(self.cat_embeddings(Variable(categories)).size())


        return self.mlp(torch.cat([hidden.squeeze(), Variable(spatials), self.cat_embeddings(Variable(categories))], 1))
