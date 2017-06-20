from Models.Encoder import Encoder
from Models.Decoder import Decoder
from Preprocessing.DataReader import DataReader

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.autograd as autograd
from torch.autograd import Variable


data_path               = "Preprocessing/Data/preprocessed.h5"
indicies_path           = "Preprocessing/Data/indices.json"
images_path             = "train2014"
images_features_path    = "Preprocessing/Data/image_features.h5"

dr = DataReader(data_path=data_path, indicies_path=indicies_path, images_path=images_path, images_features_path=images_features_path)



### Hyperparemters

# Encoder
word2index              = dr.get_word2ind()
vocab_size              = len(word2index)
word_embedding_dim      = 100
hidden_encoder_dim      = 128
visual_features_dim     = 4096

# Decoder
hidden_decoder_dim      = 128
index2word              = dr.get_ind2word()

# Training
iterations              = 10
encoder_lr              = 0.1
decoder_lr              = 0.1



encoder_model = Encoder(vocab_size, word_embedding_dim, hidden_encoder_dim, word2index, visual_features_dim)
decoder_model = Decoder(hidden_encoder_dim, hidden_decoder_dim, vocab_size)

decoder_loss_function = nn.NLLLoss()

encoder_optimizer = optim.Adam(encoder_model.parameters(), encoder_lr)
decoder_optimizer = optim.Adam(decoder_model.parameters(), decoder_lr)

decoder_epoch_loss = torch.Tensor()

game_ids = dr.get_game_ids()
game_ids = game_ids[2:5]

for epoch in range(iterations):

    for gid in game_ids:

        # check for successful training instance, else skip
        if dr.get_success(gid) == 0:
            continue

        print("Processing game", gid)

        decoder_loss = 0

        # Initiliaze encoder/decoder hidden state with 0
        encoder_model.hidden_encoder = encoder_model.init_hidden()
        decoder_model.hidden_decoder = decoder_model.init_hidden()

        encoder_model.zero_grad()
        decoder_model.zero_grad()

        questions = dr.get_questions(gid)
        visual_features = dr.get_image_features(gid)


        for qid, q in enumerate(questions):

            prod_q = str()

            if qid <= len(questions)-2:
                # more questions to come

                # encode question
                encoder_out, encoder_hidden_state = encoder_model(q, visual_features)


                # get decoder target
                next_question_length = len(questions[qid+1].split())

                decoder_targets = Variable(torch.LongTensor(next_question_length)) # TODO add -1 when -EOS- is avail.
                for qwi, qw in enumerate(questions[qid+1].split()): # TODO add [1:] slice when -SOS- is avail.
                    decoder_targets[qwi] = word2index[qw]

                # get produced question by decoder
                for next_qwid in range(next_question_length-1):
                    # go as long as target or until ?/-EOS- token

                    # pass through decoder
                    pw = decoder_model(encoder_hidden_state[0])

                    # get argmax()
                    _, w_id = pw.data.topk(1)
                    w_id = str(w_id[0][0])

                    # save produced word
                    prod_q += index2word[w_id] + ' '

                    decoder_loss += decoder_loss_function(pw, decoder_targets[next_qwid])

                    if w_id == word2index['?']: # TODO change to -EOS- once avail.
                        break


                print(prod_q)


            decoder_epoch_loss = torch.cat([decoder_epoch_loss, decoder_loss.data])

        decoder_loss.backward()

        encoder_optimizer.step()
        decoder_optimizer.step()


    print("Epoch %i, Loss %f" %(epoch, torch.mean(decoder_epoch_loss)))

print("Training completed.")