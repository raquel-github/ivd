from Models.LSTMQA import LSTMQA
from Preprocessing.DataReader import DataReader
from Preprocessing.BatchUtil2 import create_batches, get_game_ids_with_max_length, lstm_data
from pytorch_files.masked_cross_entropy import masked_cross_entropy


import numpy as np
from time import time
from random import shuffle
import datetime
import pickle
import os.path
from copy import deepcopy
import getpass

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
ts                      = str(datetime.datetime.fromtimestamp(time()).strftime('%Y_%m_%d_%H_%M'))
output_file             = "logs/lstm_output" + ts + ".log"
loss_file               = "logs/lstm_loss" + ts + ".log"
hyperparameters_file    = "logs/lstm_hyperparameters" + ts + ".log"

dr = DataReader(data_path=data_path, indicies_path=indicies_path, images_path=images_path, images_features_path=images_features_path)


### Hyperparemters
# General
my_sys                  = getpass.getuser() == 'nabi'
length                  = 11
logging                 = True if my_sys else False
save_models             = True if my_sys else False

# LSTM
word2index              = dr.get_word2ind()
index2word              = dr.get_ind2word()
vocab_size              = len(word2index)
lstm_lr					= 0.0001
word_embedding_dim      = 512
hidden_dim			    = 512
visual_features_dim     = 4096


iterations				= 100
n_games_to_train		= 96000
train_val_ratio			= 0.1
lstm_model_path     	= 'Models/bin/lstm_model'+str(n_games_to_train)+'_'+ts+'_'

if logging:
    with open(hyperparameters_file, 'a') as hyp:
        hyp.write("length %i \n" %(length))
        hyp.write("word_embedding_dim %i \n" %(word_embedding_dim))
        hyp.write("hidden_dim %i \n" %(hidden_dim))
        hyp.write("visual_features_dim %i \n" %(visual_features_dim))
        hyp.write("lstm_model_path %s \n" %(lstm_model_path))
        hyp.write("iterations %i \n" %(iterations))
        hyp.write("lstm_lr %f \n" %(lstm_lr))
        # hyp.write("grad_clip %f \n" %(grad_clip))
        hyp.write("train_val_ratio %f \n" %(train_val_ratio))
        hyp.write("save_models %f \n" %(save_models))


lstm_model = LSTMQA(vocab_size, word_embedding_dim, hidden_dim, word2index, visual_features_dim, length)

if use_cuda:
	lstm_model.cuda()

lstm_loss = nn.NLLLoss()

lstm_optimizer = optim.Adam(lstm_model.parameters(), lstm_lr)

# Get all the games which have been successful

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


for epoch in range(iterations):
    print("Epoch: ", epoch)
    start = time()
    if use_cuda:
        lstm_epoch_loss = torch.cuda.FloatTensor(1)
        lstm_epoch_loss_validation = torch.cuda.FloatTensor(1)
    else:
        lstm_epoch_loss = torch.Tensor()
        lstm_epoch_loss_validation = torch.Tensor()

    for gid in game_ids_train+game_ids_val:
    	train_game = gid in game_ids_train

    	lstm_optimizer.zero_grad()
    	lstm_model.hidden_state = lstm_model.init_hidden()

    	visual_features = torch.Tensor(dr.get_image_features(gid))

    	lstm_in, lstm_target = lstm_data(dr, gid, word2index)

    	game_loss = 0

    	for qn,(q,t) in enumerate(zip(lstm_in,lstm_target)):
    		produced_questions = ''
    		lstm_q = torch.LongTensor(q)
    
    		pred_q, _ = lstm_model(lstm_q, visual_features)

    		PQindexes = pred_q.cpu().topk(1)[1]

    		if use_cuda:
    			lstm_t = Variable(torch.LongTensor(t)).cuda()
    		else:
    			lstm_t = Variable(torch.LongTensor(t))

    		game_loss += lstm_loss(pred_q, lstm_t)

    		for pq in PQindexes.data:
    			produced_questions += ' ' + index2word[str(pq.numpy()[0])]

    		if logging:
		        if gid in [6,5000,15000,50000]:		   
		            with open(output_file, 'a') as out:		
		                out.write("%03d, %i, %i, %i, %s\n" %(epoch, gid, qn, gid in game_ids_train[::2], produced_questions))

    	if train_game:
    		lstm_epoch_loss = torch.cat([lstm_epoch_loss, game_loss.data])
    		game_loss.backward()
    		lstm_optimizer.step()
    	else:
    		lstm_epoch_loss_validation = torch.cat([lstm_epoch_loss_validation, game_loss.data])

    	if(gid%2500==0 and gid>1):
    		print("GID: "+gid+" Done")


    print("Epoch %03d, Time taken %.2f, Training-Loss %.5f, Validation-Loss %.5f" %(epoch, time()-start,torch.mean(lstm_epoch_loss), torch.mean(lstm_epoch_loss_validation)))       
    # write loss
    if logging:
        with open(loss_file, 'a') as out:
            out.write("%f, %f \n" %(torch.mean(lstm_epoch_loss), torch.mean(lstm_epoch_loss_validation)))

    if save_models:
        torch.save(lstm_model.state_dict(), lstm_model_path + str(epoch))

        print('Models saved for epoch:', epoch)