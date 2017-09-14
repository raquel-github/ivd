import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader

import json
import gzip
import collections
import os
import io
from nltk.tokenize import TweetTokenizer

import codecs

reader = codecs.getreader("utf-8")


def get_spatial_feat(bbox, im_width, im_height):

    x_width = bbox[2]
    y_height = bbox[3]

    x_left = bbox[0]
    x_right = x_left + x_width

    y_upper = im_height - bbox[1]
    y_lower = y_upper - y_height

    x_center = x_left + 0.5*x_width
    y_center = y_lower + 0.5*y_height

    # Rescale features fom -1 to 1

    x_left = (1.*x_left / im_width) * 2 - 1
    x_right = (1.*x_right / im_width) * 2 - 1
    x_center = (1.*x_center / im_width) * 2 - 1

    y_lower = (1.*y_lower / im_height) * 2 - 1
    y_upper = (1.*y_upper / im_height) * 2 - 1
    y_center = (1.*y_center / im_height) * 2 - 1

    x_width = (1.*x_width / im_width) * 2
    y_height = (1.*y_height / im_height) * 2

    # Concatenate features
    feat = [x_left, y_lower, x_right, y_upper, x_center, y_center, x_width, y_height]

    return feat

def create_data_files(data_dir='data/', dict_dir='data/', min_occ=3):
    # Set default values
    word2i = {'<padding>': 0,
              '<start>': 1,
              '<stop>': 2,
              '<stop_dialogue>': 3,
              '<unk>': 4,
              '<yes>' : 5,
              '<no>': 6,
              '<n/a>': 7,
              }

    word2occ = collections.defaultdict(int)
    categories_set = set()

    tknzr = TweetTokenizer(preserve_case=False)

    path = os.path.join(data_dir, "guesswhat.train.jsonl.gz")
    with gzip.open(path) as f:
        for k , line in enumerate(f):
            dialogue = json.loads(line.decode('utf-8'))

            for o in dialogue['objects']:
                categories_set.add(o['category'])

            for qa in dialogue['qas']:
                tokens = tknzr.tokenize(qa['question'])
                for tok in tokens:
                    word2occ[tok] += 1

    cat2i = collections.defaultdict(str)
    for c in categories_set:
        cat2i[c] = len(cat2i)

    cat_path = os.path.join(data_dir, 'cat_dict.json')
    with io.open(cat_path, 'wb') as f_out:
        data = json.dumps({'cat2i': cat2i}, ensure_ascii=False)
        f_out.write(data.encode('utf8', 'replace'))


    for word, occ in word2occ.items():
        if occ >= min_occ and word.count('.') <= 1:
            word2i[word] = len(word2i)

    dict_path = os.path.join(data_dir, 'dict.json')
    with io.open(dict_path, 'wb') as f_out:
        data = json.dumps({'word2i': word2i}, ensure_ascii=False)
        f_out.write(data.encode('utf8', 'replace'))

    questions = collections.defaultdict(int)
    answers = collections.defaultdict(int)
    categories = collections.defaultdict(int)
    spatials = collections.defaultdict(int)

    a2i = {'Yes': 1, 'No': 0, 'N/A': 2}
    with gzip.open(path) as f:
        for k , line in enumerate(f):
            dialogue = json.loads(line.decode('utf-8'))

            im_width = dialogue['image']['width']
            im_height = dialogue['image']['height']

            for o in dialogue['objects']:
                if o['id'] == dialogue['object_id']:
                    bbox = o['bbox']
                    category = o['category']
                    break

            for qa in dialogue['qas']:
                tokens = tknzr.tokenize(qa['question'])
                questions[len(questions)] = [word2i[w] if w in word2i else word2i['<unk>'] for w in tokens]
                answers[len(answers)] = a2i[qa['answer']]
                categories[len(categories)] = cat2i[category]
                spatials[len(spatials)] = get_spatial_feat(bbox=bbox, im_width=im_width, im_height=im_height)

    questions_path = os.path.join(data_dir, 'questions.json')
    with io.open(questions_path, 'wb') as f_out:
        data = json.dumps({'questions': questions}, ensure_ascii=False)
        f_out.write(data.encode('utf8', 'replace'))

    answers_path = os.path.join(data_dir, 'answers.json')
    with io.open(answers_path, 'wb') as f_out:
        data = json.dumps({'answers': answers}, ensure_ascii=False)
        f_out.write(data.encode('utf8', 'replace'))

    categories_path = os.path.join(data_dir, 'categories.json')
    with io.open(categories_path, 'wb') as f_out:
        data = json.dumps({'categories': categories}, ensure_ascii=False)
        f_out.write(data.encode('utf8', 'replace'))

    spatials_path = os.path.join(data_dir, 'spatials.json')
    with io.open(spatials_path, 'wb') as f_out:
        data = json.dumps({'spatials': spatials}, ensure_ascii=False)
        f_out.write(data.encode('utf8', 'replace'))

class OracleDataset(Dataset):
    def __init__(self, data_dir='data/'):
        super(OracleDataset, self).__init__()

        self.word2i = json.load(open("data/dict.json", 'r'))['word2i']
        self.questions = json.load(open("data/questions.json", 'r'))['questions']
        self.answers = json.load(open("data/answers.json", 'r'))['answers']
        self.categories = json.load(open("data/categories.json", 'r'))['categories']
        self.spatials = json.load(open("data/spatials.json", 'r'))['spatials']


    def __len__(self):
        return len(self.questions)

    def __getitem__(self, idx):
        padded_question = torch.LongTensor(10).fill_(self.word2i['<padding>'])

        for i,tok in enumerate(self.questions[str(idx)]):
            padded_question[i] = tok
            if i == 9:
                break

        return {'question': padded_question,
                'answer': self.answers[str(idx)],
                'category': self.categories[str(idx)],
                'spatial': torch.FloatTensor(self.spatials[str(idx)])}


# create_data_files()
"""
dataset = GuessWhatDataset()
dataloader = DataLoader(dataset=dataset, batch_size=8)
for i_batch, sample_batched in enumerate(dataloader):
    print(sample_batched['question'])
"""
