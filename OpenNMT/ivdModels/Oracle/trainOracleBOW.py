from OracleDataset import OracleDataset
from OracleBOW import Oracle

import numpy as np
from time import time
import datetime
import json
import getpass

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
import torch.autograd as autograd
from torch.autograd import Variable

use_cuda = torch.cuda.is_available()
# use_cuda = False

torch.manual_seed(1)
if use_cuda:
    torch.cuda.manual_seed_all(1)

train_file = '../../../../ivd_data/Oracle/oracle.train.json'
val_file = '../../../../ivd_data/Oracle/oracle.val.json'
test_file = '../../../../ivd_data/Oracle/oracle.test.json'
vocab_json_file = '../../../../ivd_data/Oracle/vocabOracle.json'

if selected_img_features == 'VGG':
    img_features_file = '../../../../ivd_data/img_features/image_features.h5'
    img2id_file     = '../../../../ivd_data/img_features/img_features2id.json'
    crop_features_file =  '../../../../ivd_data/img_features/crop_features.h5'
    crop2id_file    = '../../../../ivd_data/img_features/crop_features2id.json'
elif selected_img_features == 'ResNet':
    img_features_file = '../../../../ivd_data/img_features/ResNet/ResNetimage_features.h5'
    img2id_file     = '../../../../ivd_data/img_features/ResNet/ResNetimg_features2id.json'
    crop_features_file =  '../../../../ivd_data/img_features/ResNet/ResNetcrop_features.h5'
    crop2id_file    = '../../../../ivd_data/img_features/ResNet/ResNetcrop_features2id.json'

ts                      = str(datetime.datetime.fromtimestamp(time()).strftime('%Y_%m_%d_%H_%M'))
output_file             = "logs/output" + ts + ".log"
loss_file               = "logs/loss" + ts + ".log"
hyperparameters_file    = "logs/hyperparameters" + ts + ".log"

logging                 = False #if my_sys else True
save_models             = False #if my_sys else True
model_save_path            = "models/oracle_"+ts+'_'

## Hyperparamters
lr                        = 0.00001
word_embedding_dim      = 300
hidden_lstm_dim            = 128
with open(vocab_json_file) as file:
    vs = json.load(file)['word2ind']
    vocab_size = len(vs)
    del vs
iterations                = 100
batch_size                = 128
obj_cat_embedding_dim   = 512
obj_cat_size            = 91

# save hyperparameters in a file
if logging:
    with open(hyperparameters_file, 'a') as hyp:
        hyp.write("word_embedding_dim %i \n" %(word_embedding_dim))
        hyp.write("hidden_lstm_dim %i \n" %(hidden_lstm_dim))
        hyp.write("obj_cat_embedding_dim%i \n"%obj_cat_embedding_dim)
        hyp.write("iterations %i \n" %(iterations))
        hyp.write("lr %f \n" %(lr))
        hyp.write("save_models %f \n" %(save_models))
        hyp.write("batch_size %i \n" %(batch_size))

oracle_model = Oracle(word_embedding_dim, obj_cat_embedding_dim, hidden_lstm_dim, vocab_size, obj_cat_size)

if use_cuda:
    oracle_model.cuda()
    print(oracle_model)

oracle_loss_function = nn.NLLLoss()
# oracle_optimizer = optim.Adam(oracle_model.parameters(), lr)
oracle_optimizer = optim.Adadelta(oracle_model.parameters())
# oracle_optimizer = optim.SGD(oracle_model.parameters(), lr=lr,  momentum=0.5)

split_list = ['train', 'val']
json_files = [train_file, val_file]

for epoch in range(iterations):
    start = time()
    oracle_loss = 0

    if use_cuda:
        oracle_epoch_loss = torch.cuda.FloatTensor()
        oracle_val_loss = torch.cuda.FloatTensor()
    else:
        oracle_epoch_loss = torch.FloatTensor()
        oracle_val_loss = torch.FloatTensor()

    train_accuracy = 0
    val_accuracy = 0

    for split, json_data_file in zip(split_list, json_files):
        accuracy = []

        oracle_data = OracleDataset(split, json_data_file, img_features_file, img2id_file, crop_features_file, crop2id_file, vocab_json_file)

        dataloader = DataLoader(oracle_data, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)

        for i_batch, sample in enumerate(dataloader):
            question_batch, answer_batch, crop_features, image_features, spatial_batch, obj_cat_batch = \
                sample['question'], sample['answer'], sample['crop_features'], sample['img_features'], sample['spaital'], sample['obj_cat']

            oracle_optimizer.zero_grad()

            actual_batch_size = crop_features.size()[0]
            # print(i_batch)
            pred_answer = oracle_model(split, question_batch, obj_cat_batch, spatial_batch, crop_features, image_features, actual_batch_size)

            answer_batch = Variable(answer_batch)
            if use_cuda:
                answer_batch = answer_batch.cuda()

            oracle_loss = oracle_loss_function(pred_answer, answer_batch)

            pred = pred_answer.data.max(1)[1]
            correct = pred.eq(answer_batch.data).cpu().sum()

            acc = correct*100/len(answer_batch.data)
            accuracy.append(acc)

            if split == 'train':
                oracle_loss.backward()
                oracle_optimizer.step()
                oracle_epoch_loss = torch.cat([oracle_epoch_loss, oracle_loss.data])
            else:
                oracle_val_loss = torch.cat([oracle_val_loss, oracle_loss.data])

            # if i_batch % 100 == 0:
            #     print('Train Epoch: {} [{}/{} ({:.0f}%)] Accuracy: {:.03f}% '.format(epoch, i_batch * batch_size, len(oracle_data) , 100. * i_batch * batch_size/ len(oracle_data), accuracy[-1]))


        if split == 'train':
            train_accuracy = np.mean(accuracy)
        elif split == 'val':
            val_accuracy = np.mean(accuracy)



    if save_models:
        torch.save(oracle_model.state_dict(), model_save_path+str(epoch))

        print('Models saved for epoch:', epoch)

    # write loss
    if logging:
        with open(loss_file, 'a') as out:
            out.write("Epoch %03d, Time taken %.2f, Training-Loss %.5f, Validation-Loss %.5f, Training Accuracy %.5f, Validation Accuracy %.5f \n" %(epoch, time()-start, torch.mean(oracle_epoch_loss),  torch.mean(oracle_val_loss), train_accuracy, val_accuracy))

    print("Epoch %03d, Time taken %.2f, Training-Loss %.5f, Validation-Loss %.5f, Training Accuracy %.5f, Validation Accuracy %.5f" %(epoch, time()-start, torch.mean(oracle_epoch_loss), torch.mean(oracle_val_loss), train_accuracy, val_accuracy))
