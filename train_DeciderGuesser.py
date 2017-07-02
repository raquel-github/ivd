import torch
import torch.nn as nn
import torch.optim as optim
import torch.autograd as autograd
from torch.autograd import Variable

import onmt
import onmt.Markdown
import argparse
import math
import codecs
import os
import getpass
import pickle
import datetime
import numpy as np
from time import time
from random import random

from ivdModels.Guesser import Guesser
from ivdModels.Decider import Decider
from ivdModels.oracle import OracleBatch as Oracle

from DataReader import DataReader
from BCELossReg import BCELossReg
# from create_data import get_game_ids_with_max_length
# from Preprocessing.BatchUtil2 import pad_sos, get_game_ids_with_max_length

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


dr = DataReader(data_path=data_path, indicies_path=indicies_path, images_features_path=images_features_path, crop_features_path = crop_features_path)

### Hyperparamters
my_sys                  = getpass.getuser() == 'nabi'
length					= 11
logging                 = True if my_sys else False
save_models             = True if my_sys else False

# OpenNMT Parameters
opt = argparse.Namespace()
opt.batch_size 			= 1
opt.beam_size 			= 5
opt.gpu					= 0
opt.max_sent_length 	= 100
opt.replace_unk 		= True
opt.tgt					= None
opt.n_best 				= 1
opt.model 				= '../OpenNMT_Models/gw2-model_acc_76.87_ppl_3.02_e11.pt'

 # Namespace(batch_size=30, beam_size=5, dump_beam='', gpu=-1, max_sent_length=100, model='../OpenNMT_Models/gw2-model_acc_76.76_ppl_3.04_e9.pt', n_best=1, output='../OpenNMT_Models/output/1.txt', replace_unk=True, src='data/1', src_img_dir='', tgt=None, verbose=True)

# Guesser
hidden_encoder_dim		= 500
categories_length 		= dr.get_categories_length()
cat2id 					= dr.get_cat2id()
object_embedding_dim 	= 20
guesser_model_path		= 'ivdModels/bin/guessermodel'+ts+'_e'
decider_model_path		= 'ivdModels/bin/decidermodel'+ts+'_e'
min_guesser_valid		= math.inf


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
grad_clip               = 5.
train_val_ratio         = 0.1
batch_size				= 1 if my_sys else 1
n_games_to_train		= 96000 if my_sys else 20


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


# prune games
def get_game_ids_with_max_length(length):
    """ return all game ids where all questions are smaller then the given length """

    valid_games = list()

    for gid in dr.get_game_ids():
        candidate = True
        for q in dr.get_questions(gid):
            if len(q.split()) > length:
                candidate = False
                break

        if candidate:
            valid_games.append(gid)

    return valid_games


decider_model = Decider(hidden_encoder_dim)
guesser_model = Guesser(hidden_encoder_dim, categories_length+1, cat2id, object_embedding_dim)
# oracle 		  = Oracle(vocab_size, embedding_dim, categories_length+1, object_embedding_dim, hidden_dim, d_in, d_hin, d_hidden, d_hout, d_out, word2index, batch_size=1)
oracle 		  = Oracle(vocab_size, embedding_dim, categories_length+1, object_embedding_dim, hidden_dim, d_in, d_hin, d_hidden, d_hidden2, d_hidden3, d_hout, d_out, word2index, batch_size=1)

decider_loss_function = BCELossReg(ratio=0.9)
guesser_loss_function = nn.NLLLoss()

decider_optimizer = optim.Adam(decider_model.parameters(), decider_lr)
guesser_optimizer = optim.Adam(guesser_model.parameters(), guesser_lr)

if use_cuda:
	oracle.load_state_dict(torch.load(oracle_model_path))
	decider_model.cuda()
	guesser_model.cuda()
	decider_loss_function.cuda()
	print(oracle)
else:
	oracle.load_state_dict(torch.load(oracle_model_path, map_location=lambda storage, loc: storage))

opt.cuda = opt.gpu > -1
if opt.cuda:
    torch.cuda.set_device(opt.gpu)

translator = onmt.Translator(opt)

for param in oracle.parameters():
	param.requires_grad = False

# for param in translator.parameters():
# 	param.requires_grad = False


if not os.path.isfile('test_game_ids'+str(n_games_to_train)+'.p'):
    _game_ids = get_game_ids_with_max_length(length)
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


for epoch in range(iterations):
	print("Epoch: ",epoch)
	start = time()
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
	random_gids 			 = np.random.choice(len(game_ids_train+game_ids_val),4)

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

		if gid%25000==0:
			print("Game ID", gid)

		tgtBatch = [] # For OpenNMT

			### Produce Questions Until Decision To Guess
		while True:
 			# Decider here
			if question_number == 0:
				srcBatch = [['-SOS-']]
				predBatch, _, _, encStates = translator.translate(srcBatch, tgtBatch)
				srcBatch[0] += predBatch[0][0] 
				orcale_question = [' '.join(predBatch[0][0][1:-1])]
				encoder_hidden_state = Variable(encStates[-1].data, requires_grad = False)
			else:
				predBatch, _, _, encStates = translator.translate(srcBatch, tgtBatch)
				srcBatch[0] += predBatch[0][0] 
				orcale_question = [' '.join(predBatch[0][0][1:-1])]
				encoder_hidden_state = Variable(encStates[-1].data, requires_grad = False)


			decision = decider_model(encoder_hidden_state)

			if (decision.data[0,0] < 0.5) and question_number< 10:
				# print("Another Question!")
				out = oracle(orcale_question, spatial, object_class, crop_features.data, visual_features.data, num = 1)
				_, answer = out.topk(1)
				if use_cuda:
					answer = answer.cpu()
				srcBatch[0] += [id2ans[answer.data.numpy()[0][0]]]
				question_number += 1
			else:
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
			if gid in random_gids and epoch>1 and epoch%2 ==0:
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

			# nn.utils.clip_grad_norm(decider_model.parameters(), max_norm=grad_clip)
			# nn.utils.clip_grad_norm(guesser_model.parameters(), max_norm=grad_clip)

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

	if save_models and (epoch%2==0 or min_guesser_valid>torch.mean(guesser_epoch_loss_vali)):
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
	





