import numpy as np
import torch
from torch.autograd import Variable

def create_batches(game_ids_train, batch_size):
    """ returns game batches with from gloabl varibale batch size"""

    batches = np.asarray(game_ids_train)
    n_batches = int(len(game_ids_train)/batch_size)

    batches = batches[len(game_ids_train) % n_batches:]

    np.random.shuffle(batches)
    batches = batches.reshape(n_batches, batch_size)

    return batches

def create_batch_matrix(batch, dr, word2index, pad_token):
    """ given a batch, this function will create the input to the encoder for every game in the batch
    and the target for the decoder for every game in the batch """
    encoder_batch_list = [[] for i in range(len(batch))]
    decoder_batch_list = [[] for i in range(len(batch))]

    max_questions = 0

    # Step 1.
    # read in the questions and answers from the batch games
    # get the longest question
    for i, gid in enumerate(batch):
        questions           = dr.get_questions(gid)
        answers             = dr.get_answers(gid)

        # find the maximum length of question
        if max_questions < len(questions):
            max_questions = len(questions)

        # get the entire dialoge
        # start
        game_dialogue       = '-SOS- '
        question_dialogue   = str()
        for q, a in zip(questions, answers):
            game_dialogue = q + ' ' + a + ' '
            question_dialogue = q + ' '
            # convert the dialoge to indices
            game_dialogue_ids       = list()
            for w in game_dialogue.split():
                game_dialogue_ids.append(word2index[w])
            # convert the question into indicies
            question_dialogue_ids   = list()
            for w in question_dialogue.split()[1:]: # throw away -SOS- for decoder target
                question_dialogue_ids.append(word2index[w])

            encoder_batch_list[i].append(game_dialogue_ids)
            decoder_batch_list[i].append(question_dialogue_ids)

    # Step 2.
    # get the max length per question number
    # padding

    # matrix with game x question
    encoder_questions_list = [[] for i in range(max_questions)]
    # longest question per question number
    encoder_questions_maxlength = [0 for i in range(max_questions)]
    # question length for question number
    encoder_questions_list_length = [[] for i in range(max_questions)]

    for i in range(max_questions):
        for j in range(len(batch)):
            if i < len(encoder_batch_list[j]):
                encoder_questions_list[i].append(encoder_batch_list[j][i])
                encoder_questions_list_length[i].append(len(encoder_batch_list[j][i]))
                if encoder_questions_maxlength[i] < len(encoder_batch_list[j][i]):
                    encoder_questions_maxlength[i] = len(encoder_batch_list[j][i])
            else:
                encoder_questions_list[i].append([])
                encoder_questions_list_length[i].append(0)

    # padding of questions by max length of the question number
    for i, (ql, mqlen) in enumerate(zip(encoder_questions_list, encoder_questions_maxlength)):
        for j, q in enumerate(ql):
            for _ in range(mqlen-len(q)):
                encoder_questions_list[i][j].append(pad_token)


    # matrix with game x question
    decoder_questions_list = [[] for i in range(max_questions)]
    # longest question per question number
    decoder_questions_maxlength = [0 for i in range(max_questions)]
    # question length for question number
    decoder_questions_list_length = [[] for i in range(max_questions)]
    for i in range(max_questions):
        for j in range(len(batch)):
            if i < len(decoder_batch_list[j]):
                decoder_questions_list[i].append(decoder_batch_list[j][i])
                decoder_questions_list_length[i].append(len(decoder_batch_list[j][i]))
                if decoder_questions_maxlength[i] < len(decoder_batch_list[j][i]):
                    decoder_questions_maxlength[i] = len(decoder_batch_list[j][i])
            else:
                decoder_questions_list[i].append([])
                decoder_questions_list_length[i].append(0)

    # padding of questions by max length of the question number
    for i, (ql, mqlen) in enumerate(zip(decoder_questions_list, decoder_questions_maxlength)):
        for j, q in enumerate(ql):
            for _ in range(mqlen-len(q)):
                decoder_questions_list[i][j].append(pad_token)


    # Step 3.
    # convert to torch Variable so embeddings can easily be computed.

    encoder_batch_matrix = [Variable(torch.LongTensor(enc_q)).transpose(0,1) for enc_q in encoder_questions_list]
    decoder_batch_matrix = [Variable(torch.LongTensor(dec_q)).transpose(0,1) for dec_q in decoder_questions_list]


    return encoder_batch_matrix, decoder_batch_matrix

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


def preproc_game_for_batch(dr, gid, length, pad_token, word2index):
    """ """

    questions   = dr.get_questions(gid)
    n_questions = len(questions)

    answers     = dr.get_answers(gid)

    game_matrix = torch.ones((length + 1), n_questions) * pad_token

    for qid, q in enumerate(dr.get_questions(gid)):
        for wid, w in enumerate(q.split()):
            game_matrix[qid, wid] = word2index[w]

        game_matrix[qid, len(q.split())] = word2index[answers[qid]]

    return game_matrix


def create_batch(game_ids, path_to_preproc_games, length, pad_token):

    padding_vector = torch.ones(length + 1) * pad_token

    max_n_questions = -1
    for gid in game_ids:
        # load the game




from time import time
