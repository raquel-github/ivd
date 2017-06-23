from Models.Encoder import EncoderBatch as Encoder
from Models.Decoder import DecoderBatch as Decoder
from Preprocessing.DataReader import DataReader
from batchutil import create_batch_matrix, create_batches

import numpy as np
from time import time
from random import shuffle

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

dr = DataReader(data_path=data_path, indicies_path=indicies_path, images_path=images_path, images_features_path=images_features_path)


### Hyperparemters

# Encoder
word2index              = dr.get_word2ind()
vocab_size              = len(word2index)
word_embedding_dim      = 512
hidden_encoder_dim      = 512
encoder_model_path      = 'Models/bin/enc'

# Decoder
hidden_decoder_dim      = 512
index2word              = dr.get_ind2word()
visual_features_dim     = 4096
decoder_model_path      = 'Models/bin/dec'

# Training
iterations              = 100
encoder_lr              = 0.001
decoder_lr              = 0.005
grad_clip               = 5.
teacher_forcing         = False # if TRUE, the decoder input will always be the gold standard word embedding and not the preivous output
tf_decay_mode           = 'one-by-epoch-squared'
train_val_ratio         = 0.1
save_models             = False
batch_size              = 5
n_games_to_train        = 50

def get_teacher_forcing_p(epoch):
    """ return the probability of appyling teacher forcing at a given epoch """
    epoch += 1
    if tf_decay_mode == 'one-by-epoch': return 1/epoch
    if tf_decay_mode == 'one-by-epoch-squared': return 1/(epoch**2)


encoder_model = Encoder(vocab_size, word_embedding_dim, hidden_encoder_dim, word2index, batch_size)
decoder_model = Decoder(word_embedding_dim, hidden_decoder_dim, visual_features_dim, vocab_size)

if use_cuda:
    encoder_model.cuda()
    decoder_model.cuda()

decoder_loss_function = nn.NLLLoss()

encoder_optimizer = optim.Adam(encoder_model.parameters(), encoder_lr)
decoder_optimizer = optim.Adam(decoder_model.parameters(), decoder_lr)


_game_ids = dr.get_game_ids()
game_ids = list()
# get only successful games
while len(game_ids) < n_games_to_train:
    candidate = np.random.choice(_game_ids)
    if dr.get_success(candidate) == 1 and candidate not in game_ids:
        game_ids.append(candidate)


# make training validation split
game_ids_val = list(np.random.choice(game_ids, int(train_val_ratio*len(game_ids))))
game_ids_train = [gid for gid in game_ids if gid not in game_ids_val]


for epoch in range(iterations):
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

    for batch in np.vstack([batches, batches_val]):

        decoder_loss = 0
        decoder_loss_validation = 0

        # Initiliaze encoder/decoder hidden state with 0
        encoder_model.hidden_encoder = encoder_model.init_hidden()

        # Set gradientns back to 0
        encoder_optimizer.zero_grad()
        decoder_optimizer.zero_grad()


        # get the questions and the visual features of the current game
        visual_features_batch = torch.Tensor(batch_size, visual_features_dim)
        for i, gid in enumerate(batch):
            visual_features_batch[i] = torch.Tensor(dr.get_image_features(gid))


        encoder_batch_matrix, decoder_batch_matrix = create_batch_matrix(batch, dr, word2index,  word2index['-PAD-'])

        print("Encoder batch matrix 0", encoder_batch_matrix[0].size())

        encoder_out, encoder_hidden_state = encoder_model(encoder_batch_matrix[0])

        print("Encoder out", encoder_out.size())
        print("Encoder hidden", encoder_hidden_state[0].size())

        print("Decoder batch matrix 0", decoder_batch_matrix[0].size())
        print("Visual features batch 0", visual_features_batch[0].size())


        pw = decoder_model(visual_features_batch[0], encoder_hidden_state[0], torch.cat([encoder_model.sos]*batch_size))


        for qid, q in enumerate(questions):

            prod_q = str() # save the produced question here

            if qid <= len(questions)-1:
                # more questions to come

                # encode question and answer
                if qid == 0:
                    encoder_out, encoder_hidden_state = encoder_model('-SOS-')
                else:
                    enc_input = questions[qid-1] # input to encoder is previous question
                    enc_input += ' ' + answers[qid-1]
                    encoder_out, encoder_hidden_state = encoder_model(enc_input)

                # get decoder target
                question_length = len(q.split())
                if use_cuda:
                    decoder_targets = Variable(torch.LongTensor(question_length-1)).cuda() # subtract -1 to bc -SOS- is no target
                else:
                    decoder_targets = Variable(torch.LongTensor(question_length-1))

                for qwi, qw in enumerate(q.split()[1:]): # slicing [1:] to not add -SOS- to targets
                    decoder_targets[qwi] = word2index[qw]

                # get produced question by decoder
                for qwi in range(question_length-2):
                    # go as long as target or until -EOS- token

                    # pass through decoder
                    if qwi == 0:
                        # for the first word, the decoder takes the encoder hidden state and the SOS token as input
                        pw = decoder_model(visual_features, encoder_hidden_state, decoder_input=encoder_model.sos)
                    else:
                        # for all other words, the last decoder output and last decoder hidden state will be used by the model

                        # if teacher forcing = True, the input to the decoder will be the word embedding of the previous question word
                        if teacher_forcing and torch.rand(1)[0] > get_teacher_forcing_p(epoch):
                            decoder_input = encoder_model.word2embedd(q.split()[qwi-1]).view(1,1,-1)
                        else:
                            decoder_input = None

                        pw = decoder_model(visual_features, decoder_input=decoder_input)


                    # get argmax()
                    _, w_id = pw.data.topk(1)
                    w_id = w_id[0][0]


                    # save produced word
                    prod_q += index2word[str(w_id)] + ' '

                    if gid in game_ids_train:
                        decoder_loss += decoder_loss_function(pw, decoder_targets[qwi])
                    else:
                        decoder_loss_validation += decoder_loss_function(pw, decoder_targets[qwi])

                    if w_id == word2index['-EOS-']:
                        break

                """
                if epoch % 20 == 0 and gid in [3, 6, 10, 13, 17, 648]:
                    with open('output.log', 'a') as out:
                             out.write(prod_q + '\n')
                """
                if epoch % 10 == 0:
                    with open('output_tim.log', 'a') as out:
                             out.write("%03d, %i, %i, %s\n" %(epoch, gid, qid, prod_q))
                    #print(gid, prod_q)

            if gid in game_ids_train:
                decoder_epoch_loss = torch.cat([decoder_epoch_loss, decoder_loss.data])
            else:
                decoder_epoch_loss_validation = torch.cat([decoder_epoch_loss_validation, decoder_loss_validation.data])

        if gid in game_ids_train:
            # do back-prop and optimization only for training datapoints

            decoder_loss.backward()

            # clip gradients to prevent gradient explosion
            nn.utils.clip_grad_norm(encoder_model.parameters(), max_norm=grad_clip)
            nn.utils.clip_grad_norm(decoder_model.parameters(), max_norm=grad_clip)

            encoder_optimizer.step()
            decoder_optimizer.step()



    print("Epoch %03d, Time taken %.2f, Training-Loss %.5f, Validation-Loss %.5f" %(epoch, time()-start,torch.mean(decoder_epoch_loss), torch.mean(decoder_epoch_loss_validation)))

print("Training completed.")

if save_models:
    torch.save(encoder_model.state_dict(), encoder_model_path)
    torch.save(decoder_model.state_dict(), decoder_model_path)

print('Models saved.')