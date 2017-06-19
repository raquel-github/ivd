"""
The train2014 contains all images from the MS COCO training data set
For guess what, only a subset of these have been used for training.
We loop over all the images in the MS COCO set and move the images
which habe been used for GuessWhat training to a new folder.
"""

from DataReader import DataReader
import os

images_path = '../data/MS_COCO/train2014'
data_path = '../data/hdf5/preprocessed_new.h5'
indicies_path = '../data/hdf5/indices_new.json'

print(indicies_path)

dr = DataReader(data_path=data_path, indicies_path=indicies_path, images_path=images_path)

game_ids = dr.get_game_ids()

img_paths = list()
for gid in game_ids:
    img_path = dr.get_image_path(gid)
    img_paths.append(img_path)


for fn in os.listdir('../data/MS_COCO/train2014/'):
    if ('../data/MS_COCO/train2014/'+fn in img_paths):
        os.rename('../data/MS_COCO/train2014/'+fn, '../data/GuessWhat/Train/'+fn)

print('Done.')
