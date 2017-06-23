from DataReader import DataReader
import pickle as pickle

import numpy as np
"""
data_path               = "../ivd_data/preprocessed.h5"
indicies_path           = "../ivd_data/indices.json"
images_path             = "train2014"
images_features_path    = "../ivd_data/image_features.h5"

dr = DataReader(data_path=data_path, indicies_path=indicies_path, images_path=images_path, images_features_path=images_features_path)

q_lengths = list()


for gid in dr.get_game_ids():
    for q in dr.get_questions(gid):
        q_lengths.append(len(q.split()))

pickle.dump(q_lengths, open('q_lengths.p','wb'))
"""

q_lengths = pickle.load(open('q_lengths.p','rb'))

print(np.mean(q_lengths))
print(np.median(q_lengths))
print(np.std(q_lengths))
print(np.percentile(q_lengths, 80))
print(np.percentile(q_lengths, 90))
print(np.max(q_lengths))
"""
AVG     7.90230542429
MED     7.0
STD     2.44314150874
80p     9.0
90p     11.0
MAX     59
"""
