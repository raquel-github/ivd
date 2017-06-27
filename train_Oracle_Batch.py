#train_Oracle

import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from torch.autograd import Variable

import numpy as np
import h5py
from time import time

from Models.oracle import OracleBatch as Oracle
from Models.Guesser import Guesser
from Preprocessing.DataReader import DataReader
from Preprocessing.BatchUtil2 import create_batches

use_cuda = torch.cuda.is_available()

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
    max_iter = 100

    # Load data
    data_path               = "../ivd_data/preprocessed.h5"
    indicies_path           = "../ivd_data/indices.json"
    images_path             = "Preprocessing/Data/Images"
    images_features_path    = "../ivd_data/image_features.h5"   
    crop_features_path      = "../ivd_data/image_features_crops.h5" 
    dr = DataReader(data_path, indicies_path, images_path, images_features_path, crop_features_path)
    ans2id = {"Yes": 0,"No": 1,"N/A": 2}

    visual_len = 4096
    object_len = 4096 
    categories_length = dr.get_categories_length() + 1
    spatial_len = 8
    embedding_dim = 512
    word2index = dr.get_word2ind()
    index2word = dr.get_ind2word()
    vocab_size = len(word2index)
    object_embedding_dim = 20
    train_val_ratio = 0.1

    #Settings LSTM
    hidden_dim = 512
    
    #Settings MLP
    d_out = 3
    d_in = visual_len + spatial_len + object_embedding_dim + hidden_dim + object_len
    d_hin = (d_in+d_out)/4 
    d_hidden = (d_hin+d_out)/2
    d_hout = (d_hidden+d_out)/2

    batch_size = 20

    #Instance of Oracle om LSTM en MLP te runnen?
    model = Oracle(vocab_size, embedding_dim, categories_length, object_embedding_dim, hidden_dim, d_in, d_hin, d_hidden, d_hout, d_out, word2index, batch_size)
    
    # Are we using cuda?
    if use_cuda:
        model.cuda()

    # Create loss and optimizer object
    loss = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.00001)

    # Get the game IDs
    gameids = dr.get_game_ids()
    # gameids = range(250)

    # Split games in validation and training set
    gameids_val = list(np.random.choice(gameids, int(train_val_ratio*len(gameids))))
    gameids_train = [gid for gid in gameids if gid not in gameids_val]

    # Run epochs
    for epoch in range(max_iter):

        # Save start time
        start = time()

        # Create tensors for the training and validation loss
        if use_cuda:
            oracle_epoch_loss = torch.cuda.FloatTensor()
            oracle_epoch_loss_valid = torch.cuda.FloatTensor()
        else:
            oracle_epoch_loss = torch.Tensor()
            oracle_epoch_loss_valid = torch.Tensor()

        # Create the batches
        batches = create_batches(gameids_train, batch_size)
        batches_val = create_batches(gameids_val, batch_size) 
        
        # Print epoch number
        print("Epoch number %d" % (epoch))

        # Loop over batches
        for batch in np.vstack([batches, batches_val]):

            # Save statistics about the images
            corresponding_gids = []
            processed_questions = []
            processed_answers = []

            # For each question in the batch, save the answer, questions and the
            # corresponding game ID
            for x in batch:

                # Save the answer
                answers = dr.get_answers(x)

                # Parse the question
                for qi, q in enumerate(dr.get_questions(x)):
                    corresponding_gids.append(int(x))
                    processed_questions.append(" ".join(q.split()[1:-1]))
                    processed_answers.append(ans2id[answers[qi]])

            # Save the number of processed questions
            num_qas = len(processed_questions)

            # Create lists for the image, crop, spatial en object class data
            img_batch = []
            crop_batch = []
            object_class_batch = []
            spatial_batch = []

            # Loop over all processed questions
            for i in range(len(processed_questions)):
                
                # Save visual features
                img_batch.append(torch.Tensor(dr.get_image_features(corresponding_gids[i])))
                crop_batch.append(torch.Tensor(dr.get_crop_features(corresponding_gids[i])))

                # Fetch information about the objects
                objects = dr.get_object_ids(corresponding_gids[i])
                object_classes = dr.get_category_id(corresponding_gids[i])
                correct = dr.get_target_object(corresponding_gids[i])
                spatial = dr.get_image_meta(corresponding_gids[i])

                # Store information about the correct object
                for j, obj in enumerate(objects):
                    if obj == correct:
                        spatial_batch.append(img_spatial(spatial)[j])
                        object_class_batch.append(object_classes[j])
                        break

            # Pass batch on to the model
            output = model(processed_questions, spatial_batch, object_class_batch, crop_batch, img_batch, len(processed_questions))

            # Create tensor of the correct answers
            if use_cuda:
                answer = Variable(torch.LongTensor(processed_answers)).cuda()
            else:
                answer = Variable(torch.LongTensor(processed_answers))

            # Calculate the loss
            cost = loss(output, answer)

            # Check if we need to include this loss in the training or validation loss
            if batch[0] in gameids_val:
                oracle_epoch_loss_valid = torch.cat([oracle_epoch_loss, cost.data])
            else:
                oracle_epoch_loss = torch.cat([oracle_epoch_loss, cost.data])
    
            # Backpropogate Errors 
            optimizer.step()
            cost.backward()
            optimizer.zero_grad() 

        print("time:" + str(time()-start) + " \n Loss:" + str(torch.mean(oracle_epoch_loss)))
        print("Validation loss: " + str(torch.mean(oracle_epoch_loss_valid)))

        torch.save(model.state_dict(), 'Models/bin/oracle_model')


if __name__ == '__main__':
    train()
