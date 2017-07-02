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
from random import random

from Models.Encoder import EncoderBatch as Encoder
from Models.Decoder import DecoderBatch as Decoder
from Models.Guesser import Guesser
from Models.Decider import Decider
from Models.oracle import OracleBatch as Oracle

from Preprocessing.DataReader import DataReader
from Preprocessing.BatchUtil2 import pad_sos, get_game_ids_with_max_length
from BCELossReg import BCELossReg

use_cuda = torch.cuda.is_available()
# use_cuda = False

data_path               = "../ivd_data/preprocessed.h5"
indicies_path           = "../ivd_data/indices.json"
images_features_path    = "../ivd_data/image_features.h5"
crop_features_path      = "../ivd_data/image_features_crops.h5"

ts                      = str(datetime.datetime.fromtimestamp(time()).strftime('%Y_%m_%d_%H_%M'))
output_file             = "logs/deciderguesser_output" + ts + ".log"
loss_file               = "logs/decider_guesser_loss" + ts + ".log"
hyperparameters_file    = "logs/deciderguesser_hyperparameters" + ts + ".log"

dr = DataReader(data_path=data_path, indicies_path=indicies_path, images_features_path=images_features_path, crop_features_path= crop_features_path)

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
encoder_model_path      = 'Models/bin/enc2017_06_27_17_12_4'

# Decoder
hidden_decoder_dim      = 512
index2word              = dr.get_ind2word()
visual_features_dim     = 4096
decoder_model_path      = 'Models/bin/dec2017_06_27_17_12_4'
max_length              = dr.get_question_max_length()

# Guesser
categories_length 		= dr.get_categories_length()
cat2id 					= dr.get_cat2id()
object_embedding_dim 	= 20

# Oracle
word2index              = dr.get_word2ind()
vocab_size              = len(word2index)
embedding_dim 		    = 512
hidden_dim 				= 512
oracle_model_path		= '../OpenNMT_Models/oracle_optimal'
visual_len				= 4096
object_len 				= 4096 
spatial_len 			= 8
d_out 					= 3
d_in 					= visual_len + spatial_len + object_embedding_dim + hidden_dim + object_len
d_hin 					= (d_in+d_out)/2 
d_hidden 				= (d_hin+d_out)/2
d_hidden2 				= (d_hidden+d_out)/2
d_hidden3 				= (d_hidden2+d_out)/2
d_hout 					= (d_hidden3+d_out)/2
ans2id 					= {"Yes": 0,"No": 1,"N/A": 2}
id2ans					= {0: "yes", 1: "no", 2:"n/a"}

# Training
iterations              = 100
decider_lr              = 0.0001
guesser_lr              = 0.0001
grad_clip               = 50.
train_val_ratio         = 0.1
batch_size				= 1 if my_sys else 1
n_games_to_train		= 20

pad_token				= int(word2index['-PAD-'])
sos_token				= int(word2index['-SOS-'])

if logging:
    with open(hyperparameters_file, 'a') as hyp:
    	hyp.write("hidden_encoder_dim %i \n"%(hidden_encoder_dim))
    	hyp.write('categories_length %i \n' %(categories_length+1))
    	hyp.write('object_embedding_dim %i \n' %(object_embedding_dim))
    	hyp.write('vocab_size %i \n' %(vocab_size))
    	hyp.write('embedding_dim%i \n'%(embedding_dim))
    	hyp.write('hidden_dim%i \n'%(hidden_dim))
    	hyp.write('visual_len%i \n'%(visual_len))
    	hyp.write('object_len%i \n'%(object_len))
    	hyp.write('spatial_len%i \n'%(spatial_len))
    	hyp.write('id2ans: '+str(id2ans)+'\n')
    	hyp.write('iterations%i \n'%(iterations))
    	hyp.write('guesser_lr%f \n'%(guesser_lr))
    	hyp.write('decider_lr%f \n'%(decider_lr))
    	hyp.write('train_val_ratio%f \n'%(train_val_ratio))
    	hyp.write('n_games_to_train%i \n'%(n_games_to_train))


# load encoder / decoder
encoder_model = Encoder(vocab_size, word_embedding_dim, hidden_encoder_dim, word2index, visual_features_dim, length, batch_size)
decoder_model = Decoder(word_embedding_dim, hidden_decoder_dim, visual_features_dim, vocab_size, batch_size)
oracle 		  = Oracle(vocab_size, embedding_dim, categories_length+1, object_embedding_dim, hidden_dim, d_in, d_hin, d_hidden, d_hidden2, d_hidden3, d_hout, d_out, word2index, batch_size=1)

decider_loss_function = BCELossReg(ratio=0.9)
guesser_loss_function = nn.NLLLoss()


if use_cuda:
	encoder_model.load_state_dict(torch.load(encoder_model_path))
	decoder_model.load_state_dict(torch.load(decoder_model_path))
	oracle.load_state_dict(torch.load(oracle_model_path))
	decider_loss_function.cuda()
else:
	encoder_model.load_state_dict(torch.load(encoder_model_path, map_location=lambda storage, loc: storage))
	decoder_model.load_state_dict(torch.load(decoder_model_path, map_location=lambda storage, loc: storage))

# TODO Load Oracle model to play the game

for param in encoder_model.parameters():
	param.requires_grad = False

for param in decoder_model.parameters():
	param.requires_grad = False

decider_model = Decider(hidden_encoder_dim)
guesser_model = Guesser(hidden_encoder_dim, categories_length, cat2id, object_embedding_dim)

if use_cuda:
	decider_model.cuda()
	guesser_model.cuda()



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

final_no_questions_history = []
avg_no_questions_history   = []

if logging:
    with open(loss_file, 'a') as out:
    	out.write("Guesser epoch loss, Guesser epoch valid loss, Decider epoch loss, Decider valid loss, Average number of questions, Guesser Training Accuracy, Guesser Validation Accuracy ")


padded_sos = pad_sos(sos_token, pad_token, length, batch_size)
sos_embedding = encoder_model.get_sos_embedding(use_cuda)

for epoch in range(iterations):
	start = time()
	start1 = time()
	if use_cuda:
		guesser_epoch_loss 		= torch.cuda.FloatTensor()
		guesser_epoch_loss_vali = torch.cuda.FloatTensor()
		decider_epoch_loss 		= torch.cuda.FloatTensor()
		decider_epoch_loss_vali = torch.cuda.FloatTensor()
	else:
		guesser_epoch_loss 		= torch.FloatTensor()
		guesser_epoch_loss_vali = torch.FloatTensor()
		decider_epoch_loss 		= torch.FloatTensor()
		decider_epoch_loss_vali = torch.FloatTensor()

	no_questions_history = []
	guesser_wincount		 = 0
	guesser_valid_wincount 	 = 0
	random_gids 			 = np.random.choice(len(game_ids_train+game_ids_val),10)

	for gid in game_ids_train+game_ids_val:

		decider_optimizer.zero_grad()
		guesser_optimizer.zero_grad()

		guesser_loss = 0
		decider_loss = 0

		train_game = gid in game_ids_train

		if use_cuda:
			visual_features = Variable(torch.Tensor(dr.get_image_features(gid)), requires_grad=False).cuda().view(1,-1)
			crop_features	= Variable(torch.Tensor(dr.get_crop_features(gid)), requires_grad=False).cuda().view(1,-1)
		else:
			visual_features = Variable(torch.Tensor(dr.get_image_features(gid)), requires_grad=False).view(1,-1)
			crop_features	= Variable(torch.Tensor(dr.get_crop_features(gid)), requires_grad=False).view(1,-1)

		# Data required for the guesser
		img_meta 			= dr.get_image_meta(gid)
		object_categories 	= torch.LongTensor(list(map(int, dr.get_category_id(gid))))
		object_ids 			= dr.get_object_ids(gid)
		# get guesser target object
		correct_obj_id  	= dr.get_target_object(gid)
		target_guess 		= object_ids.index(correct_obj_id)

		object_spatials		= guesser_model.img_spatial(img_meta)

		spatial 			= object_spatials[target_guess]
		object_class 		= [object_categories[target_guess]]

		question_number = 0

		if gid%500==0:
			print("Game ID: "+str(gid)+" Time taken: "+str(time()-start1))
			start1 = time()

		### Produce Questions Until Decision To Guess
		while True:

			if question_number == 0:
				_, encoder_hidden_state = encoder_model(padded_sos, visual_features)
			else:
				_, encoder_hidden_state = encoder_model(seq_wid, visual_features)

			decision = decider_model(encoder_hidden_state[0])

			decision.data[0,0] = 0.4
			if (decision.data[0,0] < 0.5) and random() > 0.1:
				#print("Another Question!")
				orcale_question = ''
				seq_wid = torch.ones(length+1, batch_size, out=torch.LongTensor()) * pad_token
				for word_number in range(length):

					# decode
					if word_number == 0:
						decoder_out = decoder_model(visual_features, encoder_hidden_state, sos_embedding)
					else:
						decoder_out = decoder_model(visual_features, encoder_hidden_state)

					# sample
					w_id = torch.multinomial(torch.exp(decoder_out), 1)
					seq_wid[word_number] = w_id.data

					orcale_question += index2word[str(w_id.data[0][0])]+' '

					if index2word[str(w_id.data[0][0])] == '-EOS-' and word_number<length-2:
						orcale_question = [' '.join(orcale_question.split()[1:-1])]
						out = oracle(orcale_question, spatial, object_class, crop_features.data, visual_features.data, num = 1)
						_, answer = out.topk(1)
						if use_cuda:
							answer = answer.cpu()
						seq_wid[word_number+1] = word2index[id2ans[answer.data.numpy()[0][0]]]
						break

				question_number += 1

			else:
				#print("Guessing Time!")
				no_questions_history.append(question_number)
				break


		### Guess And Backpropagate 
		if use_cuda:
			object_spatials = Variable(object_spatials).cuda()
		else:
			object_spatials = Variable(object_spatials)

		guess = guesser_model(encoder_hidden_state, object_spatials, object_categories)
		# get best guess for decider target calc
		_, guess_id = guess.data.topk(1)
		guess_id 	= guess_id[0][0]

		if guess_id == target_guess:
			if train_game:
				guesser_wincount += 1
			else:
				guesser_valid_wincount += 1

		if logging:
			if gid in random_gids:
				with open(output_file, 'a') as out:
					out.write('epoch: '+str(epoch)+', GID: '+str(gid)+'\n')
					out.write("questions"+str(srcBatch)+'\n')
					out.write("Image URL:"+str(dr.get_image_url(gid))+'\n')
					out.write("Guess:"+str(guess_id == target_guess)+'\n')


		if use_cuda:
			guesser_loss = guesser_loss_function(guess, Variable(torch.LongTensor([target_guess])).cuda())
			decider_loss = decider_loss_function(decision, Variable(torch.Tensor([1 if guess_id == target_guess else 0])).cuda(), question_number)
		else:
			guesser_loss = guesser_loss_function(guess, Variable(torch.LongTensor([target_guess])))
			decider_loss = decider_loss_function(decision, Variable(torch.Tensor([1 if guess_id == target_guess else 0])), question_number)

		if train_game:
			guesser_loss.backward()
			decider_loss.backward()

			guesser_optimizer.step()
			decider_optimizer.step()

			guesser_epoch_loss = torch.cat([guesser_epoch_loss, guesser_loss.data])
			decider_epoch_loss = torch.cat([decider_epoch_loss, decider_loss.data])
		else:
			guesser_epoch_loss_vali = torch.cat([guesser_epoch_loss_vali, guesser_loss.data])
			decider_epoch_loss_vali = torch.cat([decider_epoch_loss_vali, decider_loss.data])

	final_no_questions_history = no_questions_history
	avg_no_questions_history.append(np.mean(no_questions_history))
	print("Epoch %03d, Time %.2f, Guesser Train Loss: %.4f, Guesser Vali Loss %.4f, Decider Train Loss %.4f, Decider Vali Loss %.4f" \
		%(epoch, time()-start,torch.mean(guesser_epoch_loss), torch.mean(guesser_epoch_loss_vali), torch.mean(decider_epoch_loss), torch.mean(decider_epoch_loss_vali)))
	print("Average number of questions %.2f, Guesser Training Accuracy %f, Guesser Validation Accuracy %f"%(np.mean(no_questions_history), guesser_wincount/len(game_ids_train), guesser_valid_wincount/len(game_ids_val)) )

	 # write loss
	if logging:
	    with open(loss_file, 'a') as out:
	        out.write("%f, %f,%f, %f,%f, %f,%f \n" %(torch.mean(guesser_epoch_loss), torch.mean(guesser_epoch_loss_vali), torch.mean(decider_epoch_loss), torch.mean(decider_epoch_loss_vali),np.mean(no_questions_history), guesser_wincount/len(game_ids_train), guesser_valid_wincount/len(game_ids_val)))

	if save_models:
	    torch.save(decider_model.state_dict(), decider_model_path + str(epoch))
	    torch.save(guesser_model.state_dict(), guesser_model_path + str(epoch))

	    if min_guesser_valid>torch.mean(guesser_epoch_loss_vali):
	    	min_guesser_valid = torch.mean(guesser_epoch_loss_vali)

	    print('Models saved for epoch:', epoch)

if logging:
	with open(loss_file, 'a') as out:
		out.write("final_no_questions_history"+ str(final_no_questions_history)+'\n')
		out.write("avg_no_questions_history"+ str(avg_no_questions_history)+'\n')
print("Training completed.")