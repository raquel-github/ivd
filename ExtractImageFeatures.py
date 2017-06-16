"""
"""

from VGG_Feature_Extract import VGG_Feature_Extract
from DataReader import DataReader
import h5py
import numpy as np

data_path = 'preprocessed_new.h5'
indicies_path = 'indices_new.json'
images_path = 'train'

dr = DataReader(data_path=data_path, indicies_path=indicies_path, images_path=images_path)
vgg_ext = VGG_Feature_Extract(images_path=images_path)

game_ids = dr.get_game_ids()

all_img_features    = np.zeros((len(game_ids), 4096))
all_img_ids         = np.zeros((len(game_ids)))

for i, gid in enumerate(game_ids):
    img_path = dr.get_image_path(gid)

    img_id = dr.get_image_id(gid)
    all_img_ids[i] = img_id

    img_features = vgg_ext.get_features(img_path)
    all_img_features[i] = img_features.data.numpy()

    if i > 5:
        break

file = h5py.File('image_features.h5', 'w')
file.create_dataset('all_img_features', dtype='float32', data=all_img_features)
file.create_dataset('all_img_ids', dtype='uint32', data=all_img_ids)
