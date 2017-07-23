from DataReader import DataReader
import numpy as np
import pickle
import os
from time import time

data_path       = '../ivd_data/preprocessed.h5'
indicies_path   = '../ivd_data/indices.json'

dr = DataReader(data_path=data_path , indicies_path=indicies_path)

# source and target files
src_train       = 'data/gw_src_train'
src_valid       = 'data/gw_src_valid'
tgt_train       = 'data/gw_tgt_train'
tgt_valid       = 'data/gw_tgt_valid'

# parameters
length          = 15
train_val_ratio = 0.1
n_games_to_train= '_ALL'
sos_token_stirng= '-SOS-'

# get all games
game_ids = dr.get_game_ids()

# prune games
def get_game_ids_with_max_length(length):
    """ return all game ids where all questions are smaller then the given length """

    valid_games = list()

    for gid in dr.get_game_ids():
        candidate = True
        for q in dr.get_questions(gid):
            if len(q.split()) > length:
                candidate = False
                break

        if candidate:
            valid_games.append(gid)

    return valid_games

if not os.path.isfile('test_game_ids'+str(n_games_to_train)+'.p'):
    _game_ids = get_game_ids_with_max_length(length)
    game_ids = list()
    # get only successful games
    for _gid in _game_ids:
        if dr.get_success(_gid) == 1:
            game_ids.append(_gid)


    pickle.dump(game_ids, open('test_game_ids'+str(n_games_to_train)+'.p', 'wb'))
else:
    game_ids = pickle.load(open('test_game_ids'+str(n_games_to_train)+'.p', 'rb'))

print('%i games loaded with max question length %i' %(len(game_ids), length))

# create training and validation set
game_ids_val    = list(np.random.choice(game_ids, int(train_val_ratio*len(game_ids))))
game_ids_train  = [gid for gid in game_ids if gid not in game_ids_val]
start = time()
tracker = 0
for gid in game_ids:

    train_game = gid in game_ids_train

    questions   = dr.get_questions(gid)
    answers     = dr.get_answers(gid)

    n_quesionts = len(questions)

    for qa_id, (question, answer) in enumerate(zip(questions, answers)):
        if qa_id == 0:
            # for the first question in the game, the source is only the sos token
            src = sos_token_stirng
        else:
            # add previous questiona and answer to the source
            src += ' ' + ' '.join(questions[qa_id-1].split()) + ' ' + answers[qa_id-1].lower()

        # target is the current question
        tgt = ' '.join(question.split())

        if train_game:
            with open(src_train, 'a') as src_train_file:
                src_train_file.write(src + '\n')
            with open(tgt_train, 'a') as tgt_train_file:
                tgt_train_file.write(tgt + '\n')

        else:
            with open(src_valid, 'a') as src_valid_file:
                src_valid_file.write(src + '\n')
            with open(tgt_valid, 'a') as tgt_valid_file:
                tgt_valid_file.write(tgt + '\n')

    tracker += 1

    if tracker % 5000 == 0 and tracker > 0:
        print('Games done: %i, Time %.2f' %(tracker, time()-start))
        start = time()
