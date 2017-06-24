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

data_path = 'Data/preprocessed.h5'
indicies_path = 'Data/indices.json'
images_path = 'Data/Images/Crops'
img_feat_path = 'Data/image_features.h5'

dr = DataReader(data_path=data_path, indicies_path=indicies_path, images_path=images_path, images_features_path=img_feat_path)
vgg_ext = VGG_Feature_Extract(images_path=images_path)

game_ids = dr.get_game_ids()

all_img_features    = np.zeros((len(game_ids), 4096))
all_img_ids         = np.zeros((len(game_ids)))

log = open("log_david_crops", 'w')

for i, gid in enumerate(game_ids):
    log.write("Image %d of %d\n" % (i, len(game_ids)))

    img_path = images_path + '/' + str(i) + '.jpg'

    img_id = dr.get_image_id(gid)
    all_img_ids[i] = img_id

    img_features = vgg_ext.get_features(img_path)
    if use_cuda:
        img_features = img_features.cpu()
    # print("Image ID\n", img_id)
    # print("Features\n",img_features.data.numpy())
    all_img_features[i] = img_features.data.numpy()

    if i > 3:
        break

file = h5py.File('Data/image_features_crops.h5', 'w')
file.create_dataset('all_img_features', dtype='float32', data=all_img_features)
file.create_dataset('all_img_ids', dtype='uint32', data=all_img_ids)

log.write("%d\n" % len(all_img_features))
log.write("Time taken: %d\n" % time()-start)

log.close()
