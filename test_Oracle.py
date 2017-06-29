import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from torch.autograd import Variable

import numpy as np
import h5py
from time import time

from Models.oracle import Oracle
from Models.Guesser import Guesser
from Preprocessing.DataReader import DataReader

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

def main():
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
    d_hin = (d_in+d_out)/2 
    d_hidden = (d_hin+d_out)/2
    d_hidden2 = (d_hidden+d_out)/2
    d_hidden3 = (d_hidden2+d_out)/2
    d_hout = (d_hidden3+d_out)/2

    batch_size = 1000

    #Instance of Oracle om LSTM en MLP te runnen?
    model = Oracle(vocab_size, embedding_dim, categories_length, object_embedding_dim, hidden_dim, d_in, d_hin, d_hidden, d_hidden2, d_hidden3, d_hout, d_out, word2index, batch_size)
    model.load_state_dict(torch.load('oracle_model_epoch_10'))

    if use_cuda:
        model.cuda()

    gameids = dr.get_game_ids()
    # print(gameids)

    # exit(0)

    correct_a = 0
    wrong = 0

    for gid in gameids:
        image = torch.Tensor(dr.get_image_features(gid))
        crop = torch.Tensor(dr.get_crop_features(gid)) 
        correct = dr.get_target_object(gid)

        objects = dr.get_object_ids(gid)
        object_class = dr.get_category_id(gid)
        spatial = dr.get_image_meta(gid) 
        for j, obj in enumerate(objects):
            if obj == correct:
                spatial = img_spatial(spatial)[j]
                object_class = object_class[j]

        quas = dr.get_questions(gid)
        answers = dr.get_answers(gid)

        for qi, question in enumerate(quas):
            outputs = model(question, spatial, object_class, crop, image)
            a_id = np.argmax(outputs.data.numpy())


            # print(answer.data.numpy())
            # print(answer.data)
            # print(ans2id[answers[qi]])

            if (a_id == ans2id[answers[qi]]):
                print("GOED")
                correct_a += 1
            else:
                print("FOUT")
                wrong += 1

            # print("=======================")

            # break
        # break
        # 
        if gid > 10:
            break

        ratio = float(correct_a) / float(correct_a + wrong)

    print("%d/%d correct answers, %d wrong, ratio: %f" % (correct_a, correct_a + wrong, wrong, ratio))

if __name__ == '__main__':
    main()
