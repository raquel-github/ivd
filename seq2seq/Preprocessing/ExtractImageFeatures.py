"""
"""

from VGG_Feature_Extract import VGG_Feature_Extract
from DataReader import DataReader
import h5py
import torch
import numpy as np
from time import time

start = time()

use_cuda = torch.cuda.is_available()

data_path = '../../data/hdf5/preprocessed.h5'
indicies_path = '../../data/hdf5/indices.json'
images_path = '../../data/GuessWhat/Train'

dr = DataReader(data_path=data_path, indicies_path=indicies_path, images_path=images_path, images_features_path= '')
vgg_ext = VGG_Feature_Extract(images_path=images_path)

game_ids = dr.get_game_ids()

all_img_features    = np.zeros((len(game_ids), 4096))
all_img_ids         = np.zeros((len(game_ids)))

for i, gid in enumerate(game_ids):
    img_path = dr.get_image_path(gid)

    img_id = dr.get_image_id(gid)
    all_img_ids[i] = img_id

    img_features = vgg_ext.get_features(img_path)
    if use_cuda:
    	img_features = img_features.cpu()
    # print("Image ID\n", img_id)
    # print("Features\n",img_features.data.numpy())
    all_img_features[i] = img_features.data.numpy()

    # if i > 15:
    #     break

file = h5py.File('image_features.h5', 'w')
file.create_dataset('all_img_features', dtype='float32', data=all_img_features)
file.create_dataset('all_img_ids', dtype='uint32', data=all_img_ids)
file.close()
print("Time taken: ", time()-start)