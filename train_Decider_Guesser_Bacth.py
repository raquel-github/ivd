import torch
import torch.nn as nn
import torch.optim as optim
import torch.autograd as autograd
from torch.autograd import Variable

import getpass
import os
import pickle
import numpy as np
from time import time

from Models.Encoder import EncoderBatch as Encoder
from Models.Decoder import DecoderBatch as Decoder
from Models.Guesser import Guesser
from Models.Decider import Decider

from Preprocessing.DataReader import DataReader
from Preprocessing.BatchUtil2 import create_batches, get_batch_visual_features, pad_sos

use_cuda = torch.cuda.is_available()


data_path               = "../ivd_data/preprocessed.h5"
indicies_path           = "../ivd_data/indices.json"
images_features_path    = "../ivd_data/image_features.h5"

dr = DataReader(data_path=data_path, indicies_path=indicies_path, images_features_path=images_features_path)

### Hyperparamters
my_sys                  = getpass.getuser() != 'nabi'
length					= 11
logging                 = False if my_sys else True
save_models             = False if my_sys else True

# Encoder
word2index              = dr.get_word2ind()
vocab_size              = len(word2index)
word_embedding_dim      = 512
hidden_encoder_dim      = 512
encoder_model_path      = 'Models/bin/enc2017_06_27_18_12_0'

# Decoder
hidden_decoder_dim      = 512
index2word              = dr.get_ind2word()
visual_features_dim     = 4096
decoder_model_path      = 'Models/bin/dec2017_06_27_18_12_0'
max_length              = dr.get_question_max_length()

# Guesser
categories_length 		= dr.get_categories_length()
cat2id 					= dr.get_cat2id()
object_embedding_dim 	= 20

# Training
iterations              = 100
decider_lr              = 0.0001
guesser_lr              = 0.0001
grad_clip               = 50.
train_val_ratio         = 0.1
batch_size				= 2 if my_sys else 200
n_games_to_train		= 20

pad_token				= int(word2index['-PAD-'])
sos_token				= int(word2index['-SOS-'])


# load encoder / decoder
encoder_model = Encoder(vocab_size, word_embedding_dim, hidden_encoder_dim, word2index, visual_features_dim, length, batch_size)
decoder_model = Decoder(word_embedding_dim, hidden_decoder_dim, visual_features_dim, vocab_size, batch_size)

if use_cuda:
	encoder_model.load_state_dict(torch.load(encoder_model_path))
	decoder_model.load_state_dict(torch.load(decoder_model_path))

else:
	encoder_model.load_state_dict(torch.load(encoder_model_path, map_location=lambda storage, loc: storage))
	decoder_model.load_state_dict(torch.load(decoder_model_path, map_location=lambda storage, loc: storage))

# TODO Load Oracle model to play the game

decider_model = Decider(hidden_encoder_dim)
guesser_model = Guesser(hidden_encoder_dim, categories_length, cat2id, object_embedding_dim)

decider_loss_function = nn.MSELoss()
guesser_loss_function = nn.NLLLoss()

decider_optimizer = optim.Adam(decider_model.parameters(), decider_lr)
guesser_optimizer = optim.Adam(guesser_model.parameters(), guesser_lr)

if not os.path.isfile('test_game_ids'+str(n_games_to_train)+'.p'):
    _game_ids = get_game_ids_with_max_length(dr, length)
    game_ids = list()
    # get only successful games
    for _gid in _game_ids:
        if dr.get_success(_gid) == 1:
            if len(game_ids) < n_games_to_train:
                game_ids.append(_gid)
            else:
                break

    pickle.dump(game_ids, open('test_game_ids'+str(n_games_to_train)+'.p', 'wb'))
else:
    game_ids = pickle.load(open('test_game_ids'+str(n_games_to_train)+'.p', 'rb'))



print("Valid game ids done. Number of valid games: ", len(game_ids))


# make training validation split
game_ids_val = list(np.random.choice(game_ids, int(train_val_ratio*len(game_ids))))
game_ids_train = [gid for gid in game_ids if gid not in game_ids_val]

padded_sos = pad_sos(sos_token, pad_token, length, batch_size)
sos_embedding = encoder_model.get_sos_embedding(use_cuda)

for epoch in range(iterations):
	if use_cuda:
		decoder_epoch_loss = torch.cuda.FloatTensor()
		decoder_epoch_loss_validation = torch.cuda.FloatTensor()
	else:
		decoder_epoch_loss = torch.Tensor()
		decoder_epoch_loss_validation = torch.Tensor()

	batches = create_batches(game_ids_train, batch_size)
	batches_val = create_batches(game_ids_val, batch_size)

	batchFlag = False
	batch_number = 0

	guesser_loss = 0
	decider_loss = 0

	for batch in np.vstack([batches, batches_val]):
		train_batch = batch in batches
		start_batch = time()

		encoder_model.hidden_encoder = encoder_model.init_hidden(train_batch)

		visual_features_batch = get_batch_visual_features(dr, batch, visual_features_dim)

		decisions = Variable(torch.ones(batch_size) * -1)

		question_number = 0
		saved_encoder_hidden_states = torch.zeros(batch_size, hidden_encoder_dim)
		saved_encoder_hidden_bool = [False] * batch_size

		while (decisions.data.numpy() < 0.5).all() == True: # check whether the decider made the decision to guess for all games

			print(batch_number, question_number)

			if question_number == 0:
				_, encoder_hidden_state = encoder_model(padded_sos, visual_features_batch)
				decisions = decider_model(encoder_hidden_state[0])
			else:
				_, encoder_hidden_state = encoder_model(seq_wid, visual_features_batch)

				for d_id, d_val in enumerate(decisions):
					if d_val < 0.5:
						decisions[d_id] = decider_model(encoder_hidden_state[0].data[0,d_id].view(1,-1))


			# save the hidden states for games where the decision to guess has been made
			for did, deci in enumerate(decisions):
				if deci > 0.5 and saved_encoder_hidden_bool[did] == False:
					saved_encoder_hidden_states[did] = encoder_hidden_state[0].data[0,did]
					saved_encoder_hidden_bool[did] = True


			seq_wid = torch.ones(length+1, batch_size, out=torch.LongTensor()) * pad_token

			for word_number in range(length):

				if word_number == 0:
					decoder_out = decoder_model(visual_features_batch, encoder_hidden_state, sos_embedding)

				batch_w_id = torch.multinomial(torch.exp(decoder_out), 1) # sample

				seq_wid[word_number] = batch_w_id.data

				# TODO add the ORACLE ANSWER

			question_number += 1


		for i, gid in enumerate(batch):
			# Data required for the guesser
			img_meta 			= dr.get_image_meta(gid)
			object_categories 	= dr.get_category_id(gid)
			object_ids 			= dr.get_object_ids(gid)
			# get guesser target object
			correct_obj_id  	= dr.get_target_object(gid)
			target_guess 		= object_ids.index(correct_obj_id)

			guess = guesser_model(saved_encoder_hidden_states[i,:], img_meta, object_categories)

			_, guess_id = guess.data.topk(1)
			guess_id 	= guess_id[0][0]

			guesser_loss += guesser_loss_function(guess,target_guess)
			decider_loss += decider_loss_function(decisions[i], 1 if guess_id == target_guess else 0)

		print(guesser_loss)
		print(decider_loss)

		guesser_loss.backward()
		decider_loss.backward()

		guesser_optimizer.step()
		decider_optimizer.step()

			#guesser_epoch_loss = torch.cat([guesser_epoch_loss, guesser_loss.data])
			#decider_epoch_loss = torch.cat([decider_epoch_loss, decider_loss.data])


		batch_number += 1


"""


for epoch in range(iterations):

	decider_epoch_loss = torch.Tensor()
	guesser_epoch_loss = torch.Tensor()

	for gid in game_ids:
		prod_q = '-SOS-'
		decision = 0

		visual_features = torch.Tensor(dr.get_image_features(gid))

		while decision < 0.5:
			encoder_outputs, encoder_hidden_state = encoder_model(prod_q)

			prod_q = str()

			decision = decider_model(encoder_hidden_state)

			# TODO: Implement Beam Search here
			if decision < 0.5:
				for qwi in range(max_length):
					# pass through decoder
					if qwi == 0:
			            # for the first word, the decoder takes the encoder hidden state and the SOS token as input
						pw = decoder_model(visual_features, encoder_hidden_state, decoder_input=encoder_model.sos)
					else:
			            # for all other words, the last decoder output and last decoder hidden state will be used by the model
						pw = decoder_model(visual_features)


			        # get argmax()
					_, w_id = pw.data.topk(1)
					w_id = str(w_id[0][0])


			        # save produced word
					prod_q += index2word[w_id] + ' '

					if w_id == word2index['-EOS-']: # TODO change to -EOS- once avail.
						break

		# Data required for the guesser
		img_meta 			= dr.get_image_meta(gid)
		object_categories 	= dr.get_category_id(gid)
		object_ids 			= dr.get_object_ids(gid)
		# get guesser target object
		correct_obj_id  	= dr.get_correct_object_id(gid)
		target_guess 		= object_ids.index(correct_obj_id)

		guess = guesser_model(encoder_hidden_state, img_meta, object_categories)

		_, guess_id = guess.data.topk(1)
		guess_id 	= guess_id[0][0]

		guesser_loss = guesser_loss_function(guess,target_guess)


		decider_loss = decider_loss_function(decision, 1 if guess_id == target_guess else 0)

		guesser_loss.backward()
		decider_loss.backward()

		guesser_optimizer.step()
		decider_optimizer.step()

		guesser_epoch_loss = torch.cat([guesser_epoch_loss, guesser_loss.data])
		decider_epoch_loss = torch.cat([decider_epoch_loss, decider_loss.data])

"""
