import torch
import torch.nn as nn
import torch.optim as optim
import torch.autograd as autograd
from torch.autograd import Variable

from Models.Encoder import EncoderBatch as Encoder
from Models.Decoder import DecoderBatch as Decoder

from Preprocessing.DataReader import DataReader
from Preprocessing.BatchUtil2 import pad_sos, get_batch_visual_features

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
encoder_model_path      = 'Models/bin/enc2017_06_28_13_28_1'

# Decoder
hidden_decoder_dim      = 512
index2word              = dr.get_ind2word()
visual_features_dim     = 4096
decoder_model_path      = 'Models/bin/dec2017_06_28_13_28_1'
max_length              = dr.get_question_max_length()

# General
length                  = 11
r                       = 3
batch_size              = 1
search_mode             = 'beam' # 'beam' or 'sample' or 'argmax'

pad_token               = int(word2index['-PAD-'])
sos_token               = int(word2index['-SOS-'])

encoder_model = Encoder(vocab_size, word_embedding_dim, hidden_encoder_dim, word2index, visual_features_dim, length, batch_size)
encoder_model.load_state_dict(torch.load(encoder_model_path, map_location=lambda storage, loc: storage))

decoder_model = Decoder(word_embedding_dim, hidden_decoder_dim, visual_features_dim, vocab_size, batch_size)
decoder_model.load_state_dict(torch.load(decoder_model_path, map_location=lambda storage, loc: storage))


# get a random game
#gid = randint(0,10000)
gid = 8514
print("Game:", gid)
print("URL:", dr.get_image_url(gid))
print("Mode: ", search_mode)
#print("Q", dr.get_questions(gid))

visual_features_batch = get_batch_visual_features(dr, [gid], visual_features_dim)

padded_sos = pad_sos(sos_token, pad_token, length, batch_size)
sos_embedding = encoder_model.get_sos_embedding(use_cuda)

encoder_model.init_hidden(train_batch=0)

stack                   = list()
sequence_words          = list()
sequence_probability    = list()
beam                    = 2
for l in range(length):
    stack.append(list())
    sequence_words.append(list())
    sequence_probability.append(list())
    if l == 0:
        # encode
        _, encoder_hidden_state = encoder_model(padded_sos, visual_features_batch)

        # decode
        word_scores = decoder_model(visual_features_batch, encoder_hidden_state, sos_embedding)
        decoder_out = decoder_model.lstm_out

        # beam
        wp, w_id = word_scores.topk(beam)

        for bid in range(beam):
            stack[l].append((encoder_hidden_state,decoder_out))
            sequence_probability[l].append(0)
            sequence_probability[l][bid]    += wp.data[0,bid]
            sequence_words[l].append('')
            sequence_words[l][bid] += ' ' + index2word[str(w_id.data[0,bid])]

    else:
        for i, (enc_hs, dec_out) in enumerate(stack[l-1]):
            _, encoder_hidden_state = encoder_model(padded_sos, visual_features_batch, hidden_state_encoder=enc_hs)

            # decode
            word_scores = decoder_model(visual_features_batch, encoder_hidden_state, decoder_input=dec_out)
            decoder_out = decoder_model.lstm_out

            # beam
            wp, w_id = word_scores.topk(beam)
            for bid in range(beam):
                stack[l].append((encoder_hidden_state,decoder_out))
                for _ in range(len(sequence_words[l-1])):
                    sequence_words[l].append(list())
                    sequence_words[l][-1] = sequence_words[l-1][bid]
                    sequence_words[l][-1] += ' ' + index2word[str(w_id.data[0,bid])]

    print(f"L: {l}")
    print(sequence_words)
    
