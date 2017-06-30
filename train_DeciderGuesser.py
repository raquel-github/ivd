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
import numpy as np
from time import time
from random import random

from Models.Guesser import Guesser
from Models.Decider import Decider

from Preprocessing.DataReader import DataReader
from Preprocessing.BatchUtil2 import pad_sos, get_game_ids_with_max_length

use_cuda = torch.cuda.is_available()
# use_cuda = False

data_path               = "../ivd_data/preprocessed.h5"
indicies_path           = "../ivd_data/indices.json"
images_features_path    = "../ivd_data/image_features.h5"

dr = DataReader(data_path=data_path, indicies_path=indicies_path, images_features_path=images_features_path)

### Hyperparamters
my_sys                  = getpass.getuser() != 'nabi'
length					= 11
logging                 = False if my_sys else True
save_models             = False if my_sys else True

# OpenNMT Parameters
opt = argparse.Namespace()
opt.batch_size 			= 1
opt.beam_size 			= 5
gpu						= 0
max_sent_length 		= 100
replace_unk 			= True
tgt						= None
model 					= '../OpenNMT_Models/gw2-model_acc_76.87_ppl_3.02_e11.pt'

 # Namespace(batch_size=30, beam_size=5, dump_beam='', gpu=-1, max_sent_length=100, model='../OpenNMT_Models/gw2-model_acc_76.76_ppl_3.04_e9.pt', n_best=1, output='../OpenNMT_Models/output/1.txt', replace_unk=True, src='data/1', src_img_dir='', tgt=None, verbose=True)


 opt.cuda = opt.gpu > -1
    if opt.cuda:
        torch.cuda.set_device(opt.gpu)