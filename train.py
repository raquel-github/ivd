from Models.Encoder import Encoder
from Models.Decider import Decider
from Models.Decoder import Decoder
from Models.Guesser import Guesser
from Preprocessing.DataReader import DataReader

import torch
import torch.nn as nn
import torch.optim as optim
import torch.autograd as autograd
import torch.autograd.Variable as Variable


data_path               = "Preprocessing/Data/preprocessed_new.h5"
indicies_path           = "Preprocessing/Data/indices_new.json"
images_path             = "train2014"
images_features_path    = "Preprocessing/Data/image_features.h5"

dr = DataReader(data_path=data_path, indicies_path=indicies_path, images_path=images_path, images_features_path=images_features_path)



### Hyperparemters

# Encoder
word2index              = dr.get_word2ind()
vocab_size              = len(word2ind)
word_embedding_dim      = 100
hidden_encoder_dim      = 128
visual_features_dim     = 4096

# Decoder
hidden_decoder_dim      = 128

# Guesser
categories_length       = dr.get_categories_length()
cat2id                  = dr.get_cat2id()

# Training
iterations              = 10
encoder_lr              = 0.001
decider_lr              = 0.001
decoder_lr              = 0.001
guesser_lr              = 0.001


encoder_model = Encoder(vocab_size, word_embedding_dim, hidden_encoder_dim, word2index, visual_features_dim)
decider_model = Decider(hidden_encoder_dim)
decoder_model = Decoder(hidden_encoder_dim, hidden_decoder_dim)
guesser_model = Guesser(hidden_encoder_dim, categories_length, cat2id)

decider_loss_function = nn.MSELoss()
decoder_loss_function = nn.NLLog()
guesser_loss_function = nn.NLLog()

encoder_optimizer = optim.Adam(encoder_model.parameters(), encoder_lr)
decider_optimizer = optim.Adam(decider_model.parameters(), decider_lr)
decoder_optimizer = optim.Adam(decoder_model.parameters(), decoder_lr)
guesser_optimizer = optim.Adam(decoder_model.parameters(), guesser_lr)

for epoch in range(iterations):

    for gid in dr.get_game_ids():

        # Set Gradients to 0
        encoder_model.zero_grad()
        decider_model.zero_grad()
        decoder_model.zero_grad()

        # Initiliaze encoder/decoder hidden state with 0
        encoder_model.hidden_encoder = encoder.init_hidden()
        decoder_model.hidden_encoder = decoder.init_hidden()

        questions = dr.get_questions(gid)
        visual_features = dr.get_visual_features(gid)

        for qid, q in enumerate(questions):

            encoder_out, encoder_hidden_state = encoder_model(q, visual_features)
            decision = decider_model(encoder_hidden_state)

            if qid != len(questions):
                # more questions to come

                # compute decider loss
                decider_loss = decider_loss_function(decision, 0)

                pw = decoder_model(encoder_hidden_state)
                # get decoder target
                decoder_targets = Variable(torch.LongTensor(len(q.split())-1))
                for qwi, qw in enumerate(q.split()[1:]):
                    decoder_targets[qwi] = word2index[qw]

                # compute decoder loss
                decoder_loss = decoder_loss_function(pw, decoder_targets)

            else:
                # last question, guess
                decider_loss_function(decision, 1)

                img_meta                = dr.get_image_meta(gid)
                img_objects             = dr.get_object_ids(gid)
                img_object_categories   = dr.get_category_id(gid)

                object_predicton = guesser_model(encoder_hidden_state, img_meta, img_object_categories)

                # get guesser target
                target_obj_id = img_objects.index(dr.get_target_object(gid))

                # compute guesser loss
                guesser_loss_function(object_predicton, target_obj_id)

                guesser_loss.backward()
                guesser_optimizer.step()


        decider_loss.backward()
        decoder_loss.backward()
        guesser_loss.backward()

        encoder_optimizer.step()
        decider_optimizer.step()
        decoder_optimizer.step()

    print("Epoch %i, Loss %f" %(epoch, loss))

print("Training completed.")
