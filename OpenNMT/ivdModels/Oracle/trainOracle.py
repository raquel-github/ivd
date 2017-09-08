from OracleDataset import OracleDataset
from Oracle import Oracle

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

from tensorboard import SummaryWriter
# run with: tensorboard --logdir runs
exp_name = "baseline_test_2"
writer = SummaryWriter('../../../logs/runs/' + exp_name)
train_batch_out = 0
valid_batch_out = 0

use_cuda = torch.cuda.is_available()
# use_cuda = False

torch.manual_seed(1)
if use_cuda:
    torch.cuda.manual_seed_all(1)

train_file = '../../../../ivd_data/Oracle/oracle.train.json'
val_file = '../../../../ivd_data/Oracle/oracle.val.json'
test_file = '../../../../ivd_data/Oracle/oracle.test.json'

img_features_file = '../../../../ivd_data/img_features/image_features.h5'
img2id_file     = '../../../../ivd_data/img_features/img_features2id.json'
crop_features_file =  '../../../../ivd_data/img_features/crop_features.h5'
crop2id_file    = '../../../../ivd_data/img_features/crop_features2id.json'
vocab_json_file = '../../../../ivd_data/Oracle/vocabOracle.json'

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

writer.add_text(u'Hyperparameter/lr', str(lr))
writer.add_text("Hyperparameter/word_embedding_dim", str(word_embedding_dim))
writer.add_text("Hyperparameter/hidden_lstm_dim", str(hidden_lstm_dim))
writer.add_text("Hyperparameter/batch_size", str(batch_size))
writer.add_text("Hyperparameter/obj_cat_embedding_dim", str(obj_cat_embedding_dim))
writer.add_text("Hyperparameter/obj_cat_size", str(obj_cat_size))


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
oracle_optimizer = optim.Adam(oracle_model.parameters(), lr)
# oracle_optimizer = optim.Adadelta(oracle_model.parameters())
# oracle_optimizer = optim.SGD(oracle_model.parameters(), lr=lr)#,  momentum=0.5)

# for p in oracle_model.parameters():
#     print(type(p))
#     print(p.size())
#     print(type(p.data))
    # print(p)

def prf(predictions, targets):
    """ computes precision, recall and f1 score """
    precision, recall, f1 = list(), list(), list()
    for a in [0,1,2]:
        true_positives  = ((predictions[predictions == a]) == (targets[predictions == a])).sum()
        false_positives = ((predictions[predictions == a]) != (targets[predictions == a])).sum()
        relatives = (targets == a).sum()

        if true_positives + false_positives == 0:
            precision.append(0)
        else:
            precision.append(true_positives / (true_positives + false_positives))

        if relatives == 0:
            recall.append(0)
        else:
            recall.append(true_positives / relatives)

        if precision[-1] == 0 or recall[-1] == 0:
            f1.append(0)
        else:
            f1.append(2 * (precision[-1] * recall[-1]) / (precision[-1] + recall[-1]))


    return precision, recall, f1


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

    oracle_epoch_precisions = list()
    oracle_epoch_recalls = list()
    oracle_epoch_f1s = list()
    oracle_val_precisions = list()
    oracle_val_recalls = list()
    oracle_val_f1s = list()

    train_accuracy = 0
    val_accuracy = 0

    for split, json_data_file in zip(split_list, json_files):
        accuracy = []

        oracle_data = OracleDataset(split, json_data_file, img_features_file, img2id_file, crop_features_file, crop2id_file, vocab_json_file)

        dataloader = DataLoader(oracle_data, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=use_cuda)


        for i_batch, sample in enumerate(dataloader):
            question_batch, answer_batch, crop_features, image_features, spatial_batch, obj_cat_batch = \
                sample['question'], sample['answer'], sample['crop_features'], sample['img_features'], sample['spaital'], sample['obj_cat']

            oracle_model.zero_grad()

            actual_batch_size = crop_features.size()[0]
            # print(i_batch)
            pred_answer = oracle_model(split, question_batch, obj_cat_batch, spatial_batch, crop_features, image_features, actual_batch_size)

            answer_batch = Variable(answer_batch)
            if use_cuda:
                answer_batch = answer_batch.cuda()

            oracle_loss = oracle_loss_function(pred_answer, answer_batch)

            pred = pred_answer.data.max(1)[1]
            predCOunt = list(pred.cpu())
            correct = pred.eq(answer_batch.data).cpu().sum()

            acc = correct*100/len(answer_batch.data)
            accuracy.append(acc)

            precisions, recalls, f1s = prf(pred, answer_batch.data)

            labelCOunt = list(answer_batch.data.cpu())

            if split == 'train':
                oracle_loss.backward()
                oracle_optimizer.step()
                # for p in oracle_model.parameters():
                #     # print('Update')
                #     # print(type(p))
                #     # print(p.size())
                #     p.data.add_(-lr, p.grad.data)

                oracle_epoch_loss = torch.cat([oracle_epoch_loss, oracle_loss.data])
                oracle_epoch_precisions.append(precisions)
                oracle_epoch_recalls.append(recalls)
                oracle_epoch_f1s.append(f1s)
            else:
                oracle_val_loss = torch.cat([oracle_val_loss, oracle_loss.data])
                oracle_val_precisions.append(precisions)
                oracle_val_recalls.append(recalls)
                oracle_val_f1s.append(f1s)

            if i_batch % 100 == 0:
                #print('Train Epoch: {} [{}/{} ({:.0f}%)] Accuracy: {:.03f}%  Loss: {:0.03f} Prediction::No: {}, Yes: {}, N/A: {}| Labels::No: {}, Yes: {}, N/A: {}'.format(epoch, i_batch * batch_size, len(oracle_data) , 100. * i_batch * batch_size/ len(oracle_data), np.mean(accuracy), torch.mean(oracle_epoch_loss), predCOunt.count(0), predCOunt.count(1), predCOunt.count(2), labelCOunt.count(0), labelCOunt.count(1), labelCOunt.count(2)))
                pass

            if split == 'train':
                writer.add_scalar("Training/Batch Accuracy", acc, train_batch_out)
                writer.add_scalar("Training/Batch Loss", oracle_loss.data[0], train_batch_out)
                writer.add_scalar("Training/Mean Batch Loss", torch.mean(oracle_epoch_loss), train_batch_out)
                writer.add_scalar("Training/Batch Precision No", precisions[0], train_batch_out)
                writer.add_scalar("Training/Batch Precision Yes", precisions[1], train_batch_out)
                writer.add_scalar("Training/Batch Precision N/A", precisions[2], train_batch_out)
                writer.add_scalar("Training/Batch Recall No", recalls[0], train_batch_out)
                writer.add_scalar("Training/Batch Recall Yes", recalls[1], train_batch_out)
                writer.add_scalar("Training/Batch Recall N/A", recalls[2], train_batch_out)
                writer.add_scalar("Training/Batch F1 No", f1s[0], train_batch_out)
                writer.add_scalar("Training/Batch F1 Yes", f1s[1], train_batch_out)
                writer.add_scalar("Training/Batch F1 N/A", f1s[2], train_batch_out)
                train_batch_out += 1

            elif split == 'val':
                writer.add_scalar("Validation/Batch Accurarcy", acc, valid_batch_acc_out)
                writer.add_scalar("Validation/Batch Loss", oracle_loss.data[0], valid_batch_acc_out)
                writer.add_scalar("Validation/Mean Batch Loss", torch.mean(oracle_epoch_loss), valid_batch_acc_out)
                writer.add_scalar("Validation/Batch Precision No", precisions[0], train_batch_out)
                writer.add_scalar("Validation/Batch Precision Yes", precisions[1], train_batch_out)
                writer.add_scalar("Validation/Batch Precision N/A", precisions[2], train_batch_out)
                writer.add_scalar("Validation/Batch Recall No", recalls[0], train_batch_out)
                writer.add_scalar("Validation/Batch Recall Yes", recalls[1], train_batch_out)
                writer.add_scalar("Validation/Batch Recall N/A", recalls[2], train_batch_out)
                writer.add_scalar("Validation/Batch F1 No", f1s[0], train_batch_out)
                writer.add_scalar("Validation/Batch F1 Yes", f1s[1], train_batch_out)
                writer.add_scalar("Validation/Batch F1 N/A", f1s[2], train_batch_out)
                valid_batch_out += 1



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
    writer.add_scalar("Training/Epoch Loss", torch.mean(oracle_epoch_loss), epoch)
    writer.add_scalar("Training/Epoch Accuracy", train_accuracy, epoch)
    writer.add_scalar("Training/Epoch Precision No", np.mean([p[0] for p in oracle_epoch_precisions]), epoch)
    writer.add_scalar("Training/Epoch Precision Yes", np.mean([p[1] for p in oracle_epoch_precisions]), epoch)
    writer.add_scalar("Training/Epoch Precision N/A", np.mean([p[2] for p in oracle_epoch_precisions]), epoch)
    writer.add_scalar("Training/Epoch Recall No", np.mean([p[0] for p in oracle_epoch_recalls]), epoch)
    writer.add_scalar("Training/Epoch Recall Yes", np.mean([p[1] for p in oracle_epoch_recalls]), epoch)
    writer.add_scalar("Training/Epoch Recall N/A", np.mean([p[2] for p in oracle_epoch_recalls]), epoch)
    writer.add_scalar("Training/Epoch F1 No", np.mean([p[0] for p in oracle_epoch_f1s]), epoch)
    writer.add_scalar("Training/Epoch F1 Yes", np.mean([p[1] for p in oracle_epoch_f1s]), epoch)
    writer.add_scalar("Training/Epoch F1 N/A", np.mean([p[2] for p in oracle_epoch_f1s]), epoch)

    writer.add_scalar("Validation/Epoch Loss", torch.mean(oracle_val_loss), epoch)
    writer.add_scalar("Validation/Epoch Accuracy", val_accuracy, epoch)
    writer.add_scalar("Validation/Epoch Precision No", np.mean([p[0] for p in oracle_val_precisions]), epoch)
    writer.add_scalar("Validation/Epoch Precision Yes", np.mean([p[1] for p in oracle_val_precisions]), epoch)
    writer.add_scalar("Validation/Epoch Precision N/A", np.mean([p[2] for p in oracle_val_precisions]), epoch)
    writer.add_scalar("Validation/Epoch Recall No", np.mean([p[0] for p in oracle_val_recalls]), epoch)
    writer.add_scalar("Validation/Epoch Recall Yes", np.mean([p[1] for p in oracle_val_recalls]), epoch)
    writer.add_scalar("Validation/Epoch Recall N/A", np.mean([p[2] for p in oracle_val_recalls]), epoch)
    writer.add_scalar("Validation/Epoch F1 No", np.mean([p[0] for p in oracle_val_f1s]), epoch)
    writer.add_scalar("Validation/Epoch F1 Yes", np.mean([p[1] for p in oracle_val_f1s]), epoch)
    writer.add_scalar("Validation/Epoch F1 N/A", np.mean([p[2] for p in oracle_val_f1s]), epoch)
