import torch
import torch.nn as nn
import torch.optim as optim
import torch.autograd as autograd
from torch.autograd import Variable


from Models.Encoder import Encoder
from Models.Decoder import Decoder
from Preprocessing.DataReader import DataReader
from Models.Guesser import Guesser



data_path               = "../ivd_data/preprocessed.h5"
indicies_path           = "../ivd_data/indices.json"
images_path             = "train2014"
images_features_path    = "../ivd_data/image_features.h5"

dr = DataReader(data_path=data_path, indicies_path=indicies_path, images_path=images_path, images_features_path=images_features_path)
"""
encoder_model = Encoder(*args, **kwargs)
encoder_model.load_state_dict(torch.load(PATH))  # TODO: Add the path

decoder_model = Decoder(*args, **kwargs)
decoder_model.load_state_dict(torch.load(PATH))  # TODO: Add the path
"""
# Load Oracle model to play the game

##Hyperparameters

# Decider
#hidden_encoder_dim 	= encoder_model.hidden_encoder_dim
hidden_encoder_dim 		= 128

# Guesser
categories_length 		= dr.get_categories_length()
cat2id 					= dr.get_cat2id()
object_embedding_dim 	= 20

# Decoder
max_length              = dr.get_question_max_length()

# Training
iterations              = 10
decider_lr              = 0.0001
guesser_lr              = 0.0001

decider_model = Decider(hidden_encoder_dim)
guesser_model = Guesser(hidden_encoder_dim, categories_length, cat2id, object_embedding_dim)

decider_loss_function = nn.MSELoss()
guesser_loss_function = nn.NLLLoss()

decider_optimizer = optim.Adam(decider_model.parameters(), decider_lr)
guesser_optimizer = optim.Adam(guesser_model.parameters(), guesser_lr)


game_ids = dr.get_game_ids()
game_ids = game_ids[2710:2715]

for epoch in range(iterations):

	decider_epoch_loss = torch.Tensor()
	guesser_epoch_loss = torch.Tensor()

    for gid in game_ids:

    	prod_q = '-SOS-'
    	decision = 0

	    while decision < 0.5:
	    	encoder_out, hidden_encoder = encoder_model(prod_q)
	    	prod_q = str()

	    	decision = decider_model(hidden_encoder)

	    	# TODO: Implement Beam Search here
	    	if decision < 0.5:
	    		for qwi in range(max_length):
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

	                if w_id == word2index['?']: # TODO change to -EOS- once avail.
	                    break

	    # Data required for the guesser
	    img_meta 			= dr.get_image_meta(gid)
	    object_categories 	= dr.get_category_id(gid)
	    object_ids 			= dr.get_object_ids(gid)
	    # get guesser target object
	    correct_obj_id  	= dr.get_correct_object_id(gid)
	    target_guess 		= object_ids.index(correct_obj_id)

	    guess = guesser_model( hidden_encoder, img_meta, object_categories)

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
