import torch
import torch.nn as nn
import torch.optim as optim
import torch.autograd as autograd
from torch.autograd import Variable

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
encoder_model_path      = 'Models/bin/enc2017_06_27_17_12_4'

# Decoder
hidden_decoder_dim      = 512
index2word              = dr.get_ind2word()
visual_features_dim     = 4096
decoder_model_path      = 'Models/bin/dec2017_06_27_17_12_4'
max_length              = dr.get_question_max_length()

# General
length                  = 11
topk                    = 1
batch_size              = 1

pad_token               = int(word2index['-PAD-'])
sos_token               = int(word2index['-SOS-'])

encoder_model = Encoder(vocab_size, word_embedding_dim, hidden_encoder_dim, word2index, visual_features_dim, length, batch_size)
encoder_model.load_state_dict(torch.load(encoder_model_path, map_location=lambda storage, loc: storage))

decoder_model = Decoder(word_embedding_dim, hidden_decoder_dim, visual_features_dim, vocab_size, batch_size)
decoder_model.load_state_dict(torch.load(decoder_model_path, map_location=lambda storage, loc: storage))


# get a random game
gid = 834
print("URL", dr.get_image_url(gid))
print("Q", dr.get_questions(gid))

visual_features_batch = get_batch_visual_features(dr, [gid], visual_features_dim)

encoder_model.init_hidden(train_batch=0)

if use_cuda:
    sos_token_id = torch.LongTensor([int(word2index['-SOS-'])]).view(1,-1).cuda()
else:
    sos_token_id = torch.LongTensor([int(word2index['-SOS-'])]).view(1,-1)

sos_embedding = encoder_model.get_sos_embedding(use_cuda)

n_questions = 5
seq_p = [ [0] * n_questions] * topk
seq_w = [ [''] * n_questions] * topk
print(seq_p)
padded_sos = pad_sos(sos_token, pad_token, length, batch_size)
for k in range(topk):
    encoder_model.init_hidden(train_batch=0)
    for n in range(n_questions):
        for l in range(length+1):
            print(l)
            if l == 0:
                print(type(visual_features_batch.data))
                _, encoder_hidden_state     = encoder_model(padded_sos, visual_features_batch)
                decoder_out                 = decoder_model(visual_features_batch, encoder_hidden_state, sos_embedding)
                wp, w_id                    = decoder_out.topk(topk)
                wp, w_id                    = wp.data.numpy()[0][k], int(w_id.data.numpy()[0][k])

            else:
                _, encoder_hidden_state     = encoder_model(torch.LongTensor([w_id]).view(1,-1))
                decoder_out                 = decoder_model(visual_features_batch, encoder_hidden_state)
                wp, w_id                    = decoder_out.topk(1)
                wp, w_id                    = wp.data.numpy()[0][0], int(w_id.data.numpy()[0][0])

                seq_p[k][n] += wp
                seq_w[k][n] += index2word[str(w_id)] + ' '

                if index2word[str(w_id)] == '-EOS-':
                    break


print(seq_p)
print(seq_w)
