import numpy as np
import torch

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
    encoder_batch_list = list()
    decoder_batch_list = list()

    for i, gid in enumerate(batch):

        questions           = dr.get_questions(gid)
        answers             = dr.get_answers(gid)

        # get the entire dialoge
        game_dialogue       = str()
        question_dialogue   = str()
        for q, a in zip(questions, answers):
            game_dialogue += q + ' ' + a + ' '
            question_dialogue += q + ' '

        # convert the dialoge to indices
        game_dialogue_ids       = list()
        for w in game_dialogue.split():
            game_dialogue_ids.append(word2index[w])

        question_dialogue_ids   = list()
        for w in question_dialogue.split():
            question_dialogue_ids.append(word2index[w])

        encoder_batch_list.append(game_dialogue_ids)
        decoder_batch_list.append(question_dialogue_ids)

    max_enc_length = 0
    max_dec_length = 0

    for eb in encoder_batch_list:
        if len(eb) > max_enc_length:
            max_enc_length = len(eb)

    for i, eb in enumerate(encoder_batch_list):
        for _ in range(max_enc_length-len(eb)):
            encoder_batch_list[i].append(pad_token)


    for db in decoder_batch_list:
        if len(db) > max_dec_length:
            max_dec_length = len(db)

    for i, db in enumerate(decoder_batch_list):
        for _ in range(max_dec_length-len(db)):
            decoder_batch_list[i].append(pad_token)

    encoder_batch_matrix = torch.Tensor(encoder_batch_list)
    decoder_batch_matrix = torch.Tensor(decoder_batch_list)

    return encoder_batch_matrix, decoder_batch_matrix


def get_padded_questions(encoder_batch_matrix, word2index):
    """
    encoder_batch_matrix is a 2-D tensor,
    where dim0 is a game,
    and dim1 is a dialoge in form of word ids
    """
    sos_token_id = word2index['-SOS-']

    encoder_batch_matrix.t()
