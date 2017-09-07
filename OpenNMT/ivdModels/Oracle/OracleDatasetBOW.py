import os
import re
import json
import h5py
import numpy as np
from collections import defaultdict
from nltk.tokenize import word_tokenize
from PIL import Image

import torch
from torch.utils.data import Dataset

class OracleDataset(Dataset):
    def __init__(self, split, json_data_file, img_features_file, img2id_file, crop_features_file, crop2id_file, vocab_json_file):
        """
        split: ['train', 'val', 'test']
        """
        with open(json_data_file) as file:
            self.questions = json.load(file)['questions']
        with open(img2id_file) as file:
            self.img2id = json.load(file)[split+'2id']
        with open(crop2id_file) as file:
            self.crop2id = json.load(file)[split+'crops2id']
        
        img_h5data = h5py.File(img_features_file, 'r')
        self.img_features = img_h5data[split+'_img_features']
        del img_h5data
        crop_h5data = h5py.File(crop_features_file, 'r')
        self.crop_features = crop_h5data[split+'_crop_features']
        del crop_h5data
        
        self.ans2id = {'no':0, 'yes':1, 'n/a':2}
        with open(vocab_json_file) as file:
            self.word2ind = json.load(file)['word2ind']
            
    def __len__(self):
        return len(self.questions)
    
    def __getitem__(self, idx):
        crop_features = self.crop_features[self.crop2id[self.questions[idx]['crop_features']]]
        img_features = self.img_features[self.img2id[self.questions[idx]['img_features']]]
        spaital = torch.FloatTensor(self.questions[idx]['spatial'])
        obj_cat = self.questions[idx]['obj_cat']
        
        raw_question = self.questions[idx]['question']
        question = (np.ones(46,'uint8')*self.word2ind['-PAD-']).tolist()
        question_words = re.findall(r'\w+', raw_question.lower())+['?']
        for wid, word in enumerate(question_words):
            if word in self.word2ind:
                question[wid] = self.word2ind[word]
            else:
                question[wid] = self.word2ind['-UNK-']
            
        question = torch.LongTensor(question)
        answer = self.ans2id[self.questions[idx]['answer'].lower()]
        
        sample = {'question':question, 'answer': answer, 'crop_features':crop_features, 'img_features':img_features,\
                  'spaital':spaital, 'obj_cat':obj_cat}
        
        return sample