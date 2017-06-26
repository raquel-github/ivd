from Models.Encoder import EncoderBatch as Encoder
from Models.Decoder import DecoderBatch as Decoder
from Preprocessing.DataReader import DataReader
from Preprocessing.BatchUtil2 import create_batches, get_game_ids_with_max_length, create_batch_from_games, get_batch_visual_features
from pytorch_files.masked_cross_entropy import masked_cross_entropy

import numpy as np
from time import time
from random import shuffle
import datetime
import pickle
import os.path
from copy import deepcopy

import torch
import torch.nn as nn
import torch.optim as optim
import torch.autograd as autograd
from torch.autograd import Variable

use_cuda = torch.cuda.is_available()

data_path               = "../ivd_data/preprocessed.h5"
indicies_path           = "../ivd_data/indices.json"
images_path             = "train2014"
images_features_path    = "../ivd_data/image_features.h5"
ts                      = str(datetime.datetime.fromtimestamp(time()).strftime('%Y%m%d%H%M'))
output_file             = "logs/output" + ts + ".log"
loss_file               = "logs/loss" + ts + ".log"
hyperparameters_file    = "logs/hyperparameters" + ts + ".log"

dr = DataReader(data_path=data_path, indicies_path=indicies_path, images_path=images_path, images_features_path=images_features_path)


### Hyperparemters
# General
length                  = 11
logging                 = True

# Encoder
word2index              = dr.get_word2ind()
vocab_size              = len(word2index)
word_embedding_dim      = 512
hidden_encoder_dim      = 512
encoder_model_path      = 'Models/bin/enc'
encoder_game_path       = 'Preprocessing/preprocessed_games/gameid2matrix_encoder.p'

# Decoder
hidden_decoder_dim      = 512 
index2word              = dr.get_ind2word()
visual_features_dim     = 4096
decoder_model_path      = 'Models/bin/dec'
decoder_game_path       = 'Preprocessing/preprocessed_games/gameid2matrix_decoder.p'

# Training
iterations              = 100
encoder_lr              = 0.0001
decoder_lr              = 0.0001
grad_clip               = 50.
teacher_forcing         = False # if TRUE, the decoder input will always be the gold standard word embedding and not the preivous output
tf_decay_mode           = 'one-by-epoch-squared'
train_val_ratio         = 0.1
save_models             = True
batch_size              = 256
n_games_to_train        = 96000

# save hyperparameters in a file
if logging:
    with open(hyperparameters_file, 'a') as hyp:
        hyp.write("length %i \n" %(length))
        hyp.write("word_embedding_dim %i \n" %(word_embedding_dim))
        hyp.write("hidden_encoder_dim %i \n" %(hidden_encoder_dim))
        hyp.write("encoder_game_path %s \n" %(encoder_game_path))
        hyp.write("hidden_decoder_dim %i \n" %(hidden_decoder_dim))
        hyp.write("visual_features_dim %i \n" %(visual_features_dim))
        hyp.write("decoder_game_path %s \n" %(decoder_game_path))
        hyp.write("iterations %i \n" %(iterations))
        hyp.write("encoder_lr %f \n" %(encoder_lr))
        hyp.write("decoder_lr %f \n" %(decoder_lr))
        hyp.write("grad_clip %f \n" %(grad_clip))
        hyp.write("teacher_forcing %i \n" %(teacher_forcing))
        hyp.write("tf_decay_mode %s \n" %(tf_decay_mode))
        hyp.write("train_val_ratio %f \n" %(train_val_ratio))
        hyp.write("save_models %f \n" %(save_models))
        hyp.write("batch_size %i \n" %(batch_size))
        hyp.write("n_games_to_train %i \n" %(n_games_to_train))

def get_teacher_forcing_p(epoch):
    """ return the probability of appyling teacher forcing at a given epoch """
    epoch += 1
    if tf_decay_mode == 'one-by-epoch': return 1/epoch
    if tf_decay_mode == 'one-by-epoch-squared': return 1/(epoch**2)


encoder_model = Encoder(vocab_size, word_embedding_dim, hidden_encoder_dim, word2index, batch_size)
decoder_model = Decoder(word_embedding_dim, hidden_decoder_dim, visual_features_dim, vocab_size, batch_size)

if use_cuda:
    encoder_model.cuda()
    decoder_model.cuda()

decoder_loss_function = nn.CrossEntropyLoss()

encoder_optimizer = optim.Adam(encoder_model.parameters(), encoder_lr)
decoder_optimizer = optim.Adam(decoder_model.parameters(), decoder_lr)

# Get all the games which have been successful

if not os.path.isfile('test_game_ids.p'):
    _game_ids = get_game_ids_with_max_length(dr, length)
    game_ids = list()
    # get only successful games
    for _gid in _game_ids:
        if dr.get_success(_gid) == 1:
            if len(game_ids) < n_games_to_train:
                game_ids.append(_gid)
            else:
                break

    pickle.dump(game_ids, open('test_game_ids.p', 'wb'))
else:
    game_ids = pickle.load(open('test_game_ids.p', 'rb'))



print("Valid game ids done. Number of valid games: ", len(game_ids))

# while len(game_ids) <= n_games_to_train:
#     candidate = np.random.choice(_game_ids)
#     if dr.get_success(candidate) == 1 and candidate not in game_ids:
#         game_ids.append(candidate)


# make training validation split
game_ids_val = list(np.random.choice(game_ids, int(train_val_ratio*len(game_ids))))
game_ids_train = [gid for gid in game_ids if gid not in game_ids_val]

for epoch in range(iterations):
    print("Epoch: ", epoch)
    start = time()
    if use_cuda:
        decoder_epoch_loss = torch.cuda.FloatTensor()
        decoder_epoch_loss_validation = torch.cuda.FloatTensor()
    else:
        decoder_epoch_loss = torch.Tensor()
        decoder_epoch_loss_validation = torch.Tensor()

    # reshuffle training batches in every epoch
    batches = create_batches(game_ids_train, batch_size)
    batches_val = create_batches(game_ids_val, batch_size) # TODO: Do entire set later

    batchFlag = False
    for batch in np.vstack([batches, batches_val]):
        train_batch = batch in batches
        start_batch = time()
        # Initiliaze encoder/decoder hidden state with 0
        encoder_model.hidden_encoder = encoder_model.init_hidden(train_batch)
        
        # get the questions and the visual features of the current game
        visual_features_batch = get_batch_visual_features(dr, batch, visual_features_dim)

        encoder_batch_matrix, decoder_batch_matrix, max_n_questions, target_lengths \
            = create_batch_from_games(dr, batch, int(word2index['-PAD-']), length, word2index, train_batch, encoder_game_path, decoder_game_path)


        for qn in range(max_n_questions):

            # Set gradientns back to 0
            encoder_optimizer.zero_grad()
            decoder_optimizer.zero_grad()
            encoder_model.hidden_encoder = encoder_model.init_hidden(train_batch)

            for qh in range(qn+1):
                encoder_out, encoder_hidden_state = encoder_model(encoder_batch_matrix[qh])


            decoder_loss = 0
            decoder_loss_vali = 0

            produced_questions = [''] * batch_size

            if train_batch:
                if use_cuda:
                    decoder_outputs = Variable(torch.zeros(length, batch_size, vocab_size)).cuda()
                else:
                    decoder_outputs = Variable(torch.zeros(length, batch_size, vocab_size))
            else:
                if use_cuda:
                    decoder_outputs = Variable(torch.zeros(length, batch_size, vocab_size), volatile=True).cuda()
                else:
                    decoder_outputs = Variable(torch.zeros(length, batch_size, vocab_size), volatile=True)

            for t in range(length):

                if t == 0:
                    if use_cuda:
                        sos_embedding = encoder_model.word_embeddings(Variable(torch.LongTensor([int(word2index['-SOS-'])]*batch_size),requires_grad=False).cuda())
                    else:
                        sos_embedding = encoder_model.word_embeddings(Variable(torch.LongTensor([int(word2index['-SOS-'])]*batch_size),requires_grad=False))

                    decoder_out = decoder_model(visual_features_batch, encoder_hidden_state, sos_embedding)

                else:
                    decoder_out = decoder_model(visual_features_batch, encoder_hidden_state)
                    decoder_outputs[t] = decoder_out

                    _, w_ids = decoder_out.topk(1)

                    if use_cuda:
                        w_ids = w_ids.cpu()
                    for i, w_id in enumerate(w_ids.data.numpy()):
                        produced_questions[i] += ' ' + index2word[str(w_id[0])]


            if train_batch:
                if use_cuda:
                    decoder_target = Variable(decoder_batch_matrix[qn]).cuda()
                else:
                    decoder_target = Variable(decoder_batch_matrix[qn])

                decoder_loss = masked_cross_entropy(decoder_outputs.transpose(0,1).contiguous(), \
                    decoder_target.transpose(0,1).contiguous(),\
                    target_lengths[qn])


                decoder_loss.backward(retain_variables=False)

                # clip gradients to prevent gradient explosion
                # nn.utils.clip_grad_norm(encoder_model.parameters(), max_norm=grad_clip)
                # nn.utils.clip_grad_norm(decoder_model.parameters(), max_norm=grad_clip)

                encoder_optimizer.step()
                decoder_optimizer.step()

                #print("Train Loss %.2f" %(decoder_loss.data[0]))

                decoder_epoch_loss = torch.cat([decoder_epoch_loss, decoder_loss.data])


            else: # vali batch
                if use_cuda:
                    decoder_target = Variable(decoder_batch_matrix[qn]).cuda()
                else:
                    decoder_target = Variable(decoder_batch_matrix[qn])

                decoder_loss_vali = masked_cross_entropy(decoder_outputs.transpose(0,1).contiguous(), \
                    decoder_target.transpose(0,1).contiguous(),\
                    target_lengths[qn])

                #print("Valid Loss %.2f" %(decoder_loss_vali.data[0]))

                decoder_epoch_loss_validation = torch.cat([decoder_epoch_loss_validation, decoder_loss_vali.data])


            if logging:
                for gid in batch:
                    if gid in [6,5000,15000,50000] and epoch>1 and epoch%2 == 0:
                        batchFlag = True
                        with open(output_file, 'a') as out:
                            out.write("%03d, %i, %i, %i, %s\n" %(epoch, gid, qn, gid in game_ids_train[::2], produced_questions[-1]))

        # del encoder_batch_matrix
        # del decoder_batch_matrix
        batchFlag = False
        print("Batchtime %f" %(time()-start_batch))



    print("Epoch %03d, Time taken %.2f, Training-Loss %.5f, Validation-Loss %.5f" %(epoch, time()-start,torch.mean(decoder_epoch_loss), torch.mean(decoder_epoch_loss_validation)))
    # write loss
    if logging:
        with open(loss_file, 'a') as out:
            out.write("%f, %f \n" %(torch.mean(decoder_epoch_loss), torch.mean(decoder_epoch_loss_validation)))

    if save_models:
        torch.save(encoder_model.state_dict(), encoder_model_path)
        torch.save(decoder_model.state_dict(), decoder_model_path)

        print('Models saved for epoch:', epoch)

print("Training completed.")


