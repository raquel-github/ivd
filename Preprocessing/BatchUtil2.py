import pickle as pickle
import torch
from torch.autograd import Variable

def create_batches(game_ids_train, batch_size):
    """ returns shuffled game batches """

    batches = np.asarray(game_ids_train)
    n_batches = int(len(game_ids_train)/batch_size)

    batches = batches[len(game_ids_train) % n_batches:]

    np.random.shuffle(batches)
    batches = batches.reshape(n_batches, batch_size)

    return batches

def get_game_ids_with_max_length(dr, length):
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


def preproc_game_for_batch_encoder(dr, gid, length, pad_token, word2index):
    """ preprocesses a game for encoder batch. Will add padding tokens at the end of every question if required.
    the first question of the enoder batch is only the SOS token
    the last question in the game questions not be considered
    """

    questions   = dr.get_questions(gid)
    n_questions = len(questions)

    answers     = dr.get_answers(gid)

    game_matrix = torch.ones(n_questions, length+1) * pad_token

    # add the SOS token at the beginning for encoder
    game_matrix[0, 0] = word2index['-SOS-']

    for qid, q in enumerate(dr.get_questions(gid)[:-1]):
        # add questions
        for wid, w in enumerate(q.split()):
            game_matrix[qid+1, wid] = word2index[w]
        # add answer
        game_matrix[qid+1, len(q.split())] = word2index[answers[qid]]

    return game_matrix

def preproc_game_for_batch_decoder(dr, gid, length, pad_token, word2index):
    """ preprocesses a game for decoder batch. Will add padding tokens at the end of every question if required. """

    questions   = dr.get_questions(gid)
    n_questions = len(questions)

    game_matrix = torch.ones(n_questions, length) * pad_token

    for qid, q in enumerate(dr.get_questions(gid)):

        for wid, w in enumerate(q.split()):
            game_matrix[qid, wid] = word2index[w]

    return game_matrix


def create_batch_from_games(dr, game_ids, pad_token, encoder_game_path, decoder_game_path):
    """ returns the padded batch"""

    gameid2matrix_encoder = pickle.load(open(encoder_game_path,'rb'))
    gameid2matrix_decoder = pickle.load(open(decoder_game_path,'rb'))

    max_n_questions = -1
    for gid in game_ids:
        if gameid2matrix_encoder[gid].size()[0] > max_n_questions:
                max_n_questions = gameid2matrix_encoder[gid].size()[0]


    # Initiliaze a list with max_n_questions items
    # then make every item in the list a torch Varibale containing the padding tokens
    encoder_batch = [0] * max_n_questions
    decoder_batch = [0] * max_n_questions
    for i, eb in enumerate(encoder_batch):
        encoder_batch[i] = Variable(torch.ones(len(game_ids), length+1) * pad_token)
        decoder_batch[i] = Variable(torch.ones(len(game_ids), length) * pad_token)


    # insert the questions into the encoder and decoder batch
    for i in range(max_n_questions):
        for j, gid in enumerate(game_ids):
            if len(gameid2matrix_encoder[gid]) > i :
                encoder_batch[i][j] = gameid2matrix_encoder[gid][i]

            if len(gameid2matrix_decoder[gid]) > i:
                decoder_batch[i][j] = gameid2matrix_decoder[gid][i]


    return encoder_batch.transpose(0,1), decoder_batch.transpose(0,1)


def get_batch_visual_features(dr, game_ids, visual_features_dim):
    """ given a list of game_ids, returns the visual features of the games """
    visual_features_batch = Variable(torch.zeros(len(game_ids), visual_features_dim))
    for i, gid in enumerate(game_ids):
        visual_features_batch[i] = torch.Tensor(dr.get_image_features(gid))

    return visual_features_batch

"""
data_path               = "../ivd_data/preprocessed.h5"
indicies_path           = "../ivd_data/indices.json"
images_path             = "train2014"
images_features_path    = "../ivd_data/image_features.h5"
length                  = 11
dr = DataReader(data_path=data_path, indicies_path=indicies_path, images_path=images_path, images_features_path=images_features_path)

word2index              = dr.get_word2ind()
pad_token               = int(word2index['-PAD-'])


game_ids = get_game_ids_with_max_length(dr, length=length)

gameid2matrix_encoder = dict()
gameid2matrix_decoder = dict()

for gid in game_ids:

    gameid2matrix_encoder[gid] = preproc_game_for_batch_encoder(dr, gid, length, pad_token, word2index)
    gameid2matrix_decoder[gid] = preproc_game_for_batch_decoder(dr, gid, length, pad_token, word2index)


pickle.dump(gameid2matrix_encoder, open('gameid2matrix_encoder.p','wb'))
pickle.dump(gameid2matrix_decoder, open('gameid2matrix_decoder.p','wb'))


gameid2matrix_encoder = pickle.load(open('../ivd_data/gameid2matrix_encoder.p','rb'))
gameid2matrix_decoder = pickle.load(open('../ivd_data/gameid2matrix_decoder.p','rb'))

game_ids = list(gameid2matrix_decoder.keys())[:3]
print(get_batch_visual_features(dr, game_ids, 4096))
#create_batch_from_games(dr, game_ids, pad_token, '../ivd_data/gameid2matrix_encoder.p', '../ivd_data/gameid2matrix_decoder.p')
"""
