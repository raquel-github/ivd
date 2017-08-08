import os
import glob
import h5py
import json
import torch
import numpy as np
from time import time

from VGG_Feature_Extract import VGG_Feature_Extract

def extract_img_features(dir_path, img_list, vgg_ext, use_cuda):
	img_features = np.zeros((len(img_list), 4096))
	name2id = {}

	for idx, img in enumerate(img_list):
		img_path = dir_path+img
		name2id[img] = idx

		features = vgg_ext.get_features(img_path)
		if use_cuda:
			features = features.cpu()
		img_features[idx] = features.data.numpy()

	return img_features, name2id
 

start = time()

use_cuda = torch.cuda.is_available()

train_img = '/home/aashish/Documents/ProjectAI/data/GuessWhat/Train/'
val_img   = '/home/aashish/Documents/ProjectAI/data/GuessWhat/Validation/'
test_img  = '/home/aashish/Documents/ProjectAI/data/GuessWhat/Test/'

train_crops = '/home/aashish/Documents/ProjectAI/data/GuessWhat/Train_Crops/'
val_crops   = '/home/aashish/Documents/ProjectAI/data/GuessWhat/Validation_Crops/'
test_crops  = '/home/aashish/Documents/ProjectAI/data/GuessWhat/Test_Crops/'

vgg_ext = VGG_Feature_Extract()

train_img_features, train2id = extract_img_features(train_img, os.listdir(train_img), vgg_ext, use_cuda) 
val_img_features, val2id = extract_img_features(val_img, os.listdir(val_img), vgg_ext, use_cuda)
test_img_features, test2id = extract_img_features(test_img, os.listdir(test_img), vgg_ext, use_cuda)

img_file = h5py.File('image_features.h5', 'w')
img_file.create_dataset('train_img_features', dtype='float32', data=train_img_features)
img_file.create_dataset('val_img_features', dtype='float32', data=val_img_features)
img_file.create_dataset('test_img_features', dtype='float32', data=test_img_features)
img_file.close()

json_data = {'train2id':train2id, 'val2id':val2id, 'test2id':test2id}
with open('img_features2id.json', 'a') as file:
	json.dump(json_data, file)

print('Image Features extracted. Doing the crops now')
print('Time taken: ', time()-start)
del train_img_features, val_img_features, test_img_features, train2id, val2id, test2id

start = time()
print('Begin with crops')

train_crop_features, traincrops2id = extract_img_features(train_crops, os.listdir(train_crops), vgg_ext, use_cuda)
val_crop_features, valcrops2id = extract_img_features(val_crops, os.listdir(val_crops), vgg_ext, use_cuda)
test_crop_features, testcrops2id = extract_img_features(test_crops, os.listdir(test_crops), vgg_ext, use_cuda)

crop_file = h5py.File('crop_features.h5', 'w')
crop_file.create_dataset('train_crop_features', dtype='float32', data=train_crop_features)
crop_file.create_dataset('val_crop_features', dtype='float32', data=val_crop_features)
crop_file.create_dataset('test_crop_features', dtype='float32', data=test_crop_features)
crop_file.close()

json_data = {'traincrops2id':traincrops2id, 'valcrops2id':valcrops2id, 'testcrops2id':testcrops2id} 
with open('crop_features2id.json', 'a') as file:
	json.dump(json_data, file)

print('Crop features extracted')
print('Time taken:', time()-start)
