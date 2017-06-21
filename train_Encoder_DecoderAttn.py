from Models.Encoder import Encoder
from Models.DecoderAttn import DecoderAttn
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
word_embedding_dim      = 128
hidden_encoder_dim      = 128
visual_features_dim     = 4096

# Decoder
hidden_decoder_dim      = 128
index2word              = dr.get_ind2word()
max_length              = dr.get_question_max_length()

# Training
iterations              = 10
encoder_lr              = 0.01
decoder_lr              = 0.01



encoder_model = Encoder(vocab_size, word_embedding_dim, hidden_encoder_dim, word2index, visual_features_dim)
decoder_model = DecoderAttn(hidden_encoder_dim, hidden_decoder_dim, vocab_size, word_embedding_dim, max_length)


decoder_loss_function = nn.NLLLoss()

encoder_optimizer = optim.Adam(encoder_model.parameters(), encoder_lr)
decoder_optimizer = optim.Adam(decoder_model.parameters(), decoder_lr)


game_ids = dr.get_game_ids()
game_ids = game_ids[2710:2715]

for epoch in range(iterations):

    decoder_epoch_loss = torch.Tensor()

    for gid in game_ids:

        # check for successful training instance, else skip
        if dr.get_success(gid) == 0:
            continue

        print("Processing game", gid)

        decoder_loss = 0

        # Initiliaze encoder hidden state with 0
        encoder_model.hidden_encoder = encoder_model.init_hidden()

        # Set gradientns back to 0
        encoder_optimizer.zero_grad()
        decoder_optimizer.zero_grad()

        # get the questions and the visual features of the current game
        questions = dr.get_questions(gid)
        answers = dr.get_answers(gid)
        visual_features = dr.get_image_features(gid)


        for qid, q in enumerate(questions):

            prod_q = str() # save the produced question here

            if qid <= len(questions)-1:
                # more questions to come

                # encode question word by word and save each encoder output
                encoder_outputs = Variable(torch.zeros(max_length, hidden_encoder_dim))
                if qid == 0:
                    encoder_outputs[0], encoder_hidden_state = encoder_model('-SOS-', visual_features)
                else:
                    enc_inputq = questions[qid-1]
                    # add answer
                    enc_input += ' ' + answers[qid-1]
                    for qwi, qw in enumerate(enc_input.split()):
                        encoder_outputs[qwi] , encoder_hidden_state = encoder_model(qw, visual_features)


                # get decoder target
                question_length = len(q.split())
                decoder_targets = Variable(torch.LongTensor(question_length)) # TODO add -1 when -EOS- is avail.
                for qwi, qw in enumerate(q.split()): # TODO add [1:] slice when -SOS- is avail.
                    decoder_targets[qwi] = word2index[qw]


                # get produced question by decoder
                for qwi in range(question_length-1):
                    # go as long as target or until ?/-EOS- token

                    # pass through decoder
                    if qwi == 0:
                        # for the first word, the decoder takes the encoder hidden state and the SOS token as input
                        pw = decoder_model(encoder_outputs, encoder_hidden_state, encoder_model.sos)
                    else:
                        # for all other words, the last decoder output and last decoder hidden state will be used by the model
                        pw = decoder_model(encoder_outputs)


                    # get argmax()
                    _, w_id = pw.data.topk(1)
                    w_id = str(w_id[0][0])


                    # save produced word
                    prod_q += index2word[w_id] + ' '

                    decoder_loss += decoder_loss_function(pw, decoder_targets[qwi])

                    if w_id == word2index['?']: # TODO change to -EOS- once avail.
                        break


                print(prod_q)


            decoder_epoch_loss = torch.cat([decoder_epoch_loss, decoder_loss.data])

        decoder_loss.backward()

        encoder_optimizer.step()
        decoder_optimizer.step()


    print("Epoch %i, Loss %f" %(epoch, torch.mean(decoder_epoch_loss)))

print("Training completed.")
