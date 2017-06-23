#train_Oracle

import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from torch.autograd import Variable

import numpy
import h5py
from time import time

from Models.oracle import Oracle
from Preprocessing.DataReader import DataReader
from Models.Guesser import Guesser

def img_spatial(img_meta):
        """ returns the spatial information of a bounding box """
        bboxes = img_meta[0] # gets all bboxes in the image
        width = img_meta[1]
        height = img_meta[2]
        image_center_x = width / 2
        image_center_y = height / 2

        spatial = Variable(torch.FloatTensor(len(bboxes), 8))

        for i, bbox in enumerate(bboxes):
            x_min = bbox[0] / width
            y_min = bbox[1] / height
            x_max = (bbox[0] + bbox[2]) / width
            y_max = (bbox[1] + bbox[3]) / height

            w_box = x_max - x_min
            h_box = y_max - y_min

            x_min = x_min * 2 - 1
            y_min = y_min * 2 - 1
            x_max = x_max * 2 - 1
            y_max = y_max * 2 - 1

            x_center = (x_min + x_max) / 2
            y_center = (y_min + y_max) / 2


            spatial[i] = torch.FloatTensor([x_min, y_min, x_max, y_max, x_center, y_center, w_box, h_box])
        return spatial

def train():
    max_iter = 10 

    # Load data
    data_path               = "Preprocessing/Data/preprocessed.h5"
    indicies_path           = "Preprocessing/Data/indices.json"
    images_path             = "Preprocessing/Data/Images"
    images_features_path    = "Preprocessing/Data/image_features.h5"   
    crop_features_path      = "Preprocessing/Data/image_features_crops.h5" 
    dr = DataReader(data_path, indicies_path, images_path, images_features_path, crop_features_path)
    ans2id = {"Yes": 0,"No": 1,"N/A": 2}

    visual_len = 4096
    object_len = 4096 
    categories_length = dr.get_categories_length()
    spatial_len = 8
    embedding_dim = 128
    word2index = dr.get_word2ind()
    vocab_size = len(word2index)
    object_embedding_dim = 20

    #Settings LSTM
    hidden_dim = 128
    
    #Settings MLP
    d_out = 3
    d_in = visual_len + spatial_len + object_embedding_dim + hidden_dim + object_len
    d_hin = (d_in+d_out)/4 
    d_hidden = (d_hin+d_out)/2
    d_hout = (d_hidden+d_out)/2

    #Instance of Oracle om LSTM en MLP te runnen?
    model = Oracle(vocab_size, embedding_dim, categories_length, object_embedding_dim, hidden_dim, d_in, d_hin, d_hidden, d_hout, d_out, word2index)
    loss = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.00001)

    #Get Game/Question and run model
    gameids = range(10)#dr.get_game_ids()
    for epoch in range(max_iter):
        start = time()
        oracle_epoch_loss = torch.Tensor()
        
        print("Epoch number %d" % (epoch))
        for gid in gameids:
            image = torch.Tensor(dr.get_image_features(gid))
            crop = torch.Tensor(dr.get_crop_features(gid)) 
            correct = dr.get_target_object(gid)

            objects = dr.get_object_ids(gid)
            object_class = dr.get_category_id(gid)
            spatial = dr.get_image_meta(gid) 
            for j, obj in enumerate(objects):
                if obj == correct:
                    #print("found correct object")
                    spatial = img_spatial(spatial)[j]
                    object_class = object_class[j]

            quas = dr.get_questions(gid)
            answers = dr.get_answers(gid)
            for qi,question in enumerate(quas):
                outputs = model(question, spatial, object_class, crop, image)

                answer = Variable(torch.LongTensor([ans2id[answers[qi]]]))
                cost = loss(outputs,answer)
                oracle_epoch_loss = torch.cat([oracle_epoch_loss,cost.data])
    
                # Backpropogate Errors 
                optimizer.zero_grad() 
                cost.backward()
                optimizer.step()
        print("time:" + str(time()-start) + " \n Loss:" + str(torch.mean(oracle_epoch_loss)))
if __name__ == '__main__':
    train()
