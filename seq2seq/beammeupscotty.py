import torch
import torch.nn as nn
import torch.optim as optim
import torch.autograd as autograd
from torch.autograd import Variable

from random import random, randint

from Models.Encoder import EncoderBatch as Encoder
from Models.Decoder import DecoderBatch as Decoder

from Preprocessing.DataReader import DataReader
from Preprocessing.BatchUtil2 import get_batch_visual_features
from Preprocessing.BatchUtil2 import pad_sos

use_cuda = torch.cuda.is_available()

data_path               = "../ivd_data/preprocessed.h5"
indicies_path           = "../ivd_data/indices.json"
images_features_path    = "../ivd_data/image_features.h5"

dr = DataReader(data_path=data_path, indicies_path=indicies_path, images_features_path=images_features_path)

### Hyperparamters

# Encoder
word2index              = dr.get_word2ind()
vocab_size              = len(word2index)
word_embedding_dim      = 512
hidden_encoder_dim      = 512
encoder_model_path      = 'Models/bin/enc2017_06_28_13_28_1 copy'

# Decoder
hidden_decoder_dim      = 512
index2word              = dr.get_ind2word()
visual_features_dim     = 4096
decoder_model_path      = 'Models/bin/dec2017_06_28_13_28_1 copy'
max_length              = dr.get_question_max_length()

# General
length                  = 11
topk                    = 3
batch_size              = 1
search_mode             = 'argmax' # 'beam' or 'sample' or 'argmax'

pad_token               = int(word2index['-PAD-'])
sos_token               = int(word2index['-SOS-'])

encoder_model = Encoder(vocab_size, word_embedding_dim, hidden_encoder_dim, word2index, visual_features_dim, length, batch_size)
encoder_model.load_state_dict(torch.load(encoder_model_path, map_location=lambda storage, loc: storage))

decoder_model = Decoder(word_embedding_dim, hidden_decoder_dim, visual_features_dim, vocab_size, batch_size)
decoder_model.load_state_dict(torch.load(decoder_model_path, map_location=lambda storage, loc: storage))


# get a random game
gid = randint(0,10000)
gid = 11
print("Game:", gid)
print("URL:", dr.get_image_url(gid))
print("Mode: ", search_mode)
#print("Q", dr.get_questions(gid))

visual_features_batch = get_batch_visual_features(dr, [gid], visual_features_dim)

encoder_model.init_hidden(train_batch=0)

if use_cuda:
    sos_token_id = torch.LongTensor([int(word2index['-SOS-'])]).view(1,-1).cuda()
else:
    sos_token_id = torch.LongTensor([int(word2index['-SOS-'])]).view(1,-1)

sos_embedding = encoder_model.get_sos_embedding(use_cuda)

n_questions = 5
seq_p = []
seq_w = []

padded_sos = pad_sos(sos_token, pad_token, length, batch_size)

for k in range(topk):
    #print("+***************************")
    seq_p.append([])
    seq_w.append([])
    encoder_model.init_hidden(train_batch=0)
    #print("Init hidden state")

    for n in range(n_questions):
        #print("##################")
        seq_p[-1].append(0)
        seq_w[-1].append('')

        if n == 0:
            _, encoder_hidden_state = encoder_model(padded_sos, visual_features_batch)
            #print("Encoding sos")
        else:
            enc_in = torch.ones(length+1, 1, out=torch.LongTensor()) * pad_token
            for i,w in enumerate(seq_w[k][n-1].split()):
                enc_in[i] = word2index[w]

            _, encoder_hidden_state = encoder_model(enc_in, visual_features_batch)
            #print("encoding previous q/a")

        for l in range(length):
            #print(k,n,l)

            #print("decoding word")

            if l == 0:
                decoder_out = decoder_model(visual_features_batch, encoder_hidden_state, sos_embedding)
            else:
                decoder_out = decoder_model(visual_features_batch, encoder_hidden_state)

            if search_mode == 'sample':
                w_id    = torch.multinomial(torch.exp(decoder_out), 1)
                wp      = decoder_out.data[0,w_id.data[0,0]]
                seq_p[k][n] += wp

            elif search_mode == 'argmax':
                wp, w_id = decoder_out.topk(1)
                seq_p[k][n] += wp.data[0,0]

            #print(index2word[str(w_id.data[0,0])])
            #print(w_id.data[0,0])
            seq_w[k][n] += index2word[str(w_id.data[0,0])] + ' '

            if index2word[str(w_id.data[0,0])] == '-EOS-':
                # add answer
                if random() <= 0.7:
                    seq_w[k][n] += 'No'
                else:
                    seq_w[k][n] += 'Yes'
                break


print(seq_p)
print(seq_w)
print(len(seq_w))
