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
    encoder_batch_list = [[] for i in range(len(batch))]
    decoder_batch_list = [[] for i in range(len(batch))]

    max_questions = 0

    for i, gid in enumerate(batch):
        questions           = dr.get_questions(gid)
        answers             = dr.get_answers(gid)

        if max_questions < len(questions):
            max_questions = len(questions)

        # get the entire dialoge
        game_dialogue       = str()
        question_dialogue   = str()
        for q, a in zip(questions, answers):
            game_dialogue = q + ' ' + a + ' '
            question_dialogue = q + ' '
            # convert the dialoge to indices
            game_dialogue_ids       = list()
            for w in game_dialogue.split():
                game_dialogue_ids.append(word2index[w])
            question_dialogue_ids   = list()
            for w in question_dialogue.split():
                question_dialogue_ids.append(word2index[w])
            encoder_batch_list[i].append(game_dialogue_ids)
            decoder_batch_list[i].append(question_dialogue_ids)

    encoder_questions_list = [[] for i in range(max_questions)]
    encoder_questions_maxlength = [[0] for i in range(max_questions)]

    for i in range(max_questions):
        for j in range(len(batch)):
            if i < len(encoder_batch_list[j]):
                encoder_questions_list[i].append(encoder_batch_list[j][i])
                if encoder_questions_maxlength[i][0] < len(encoder_batch_list[j][i]):
                    encoder_questions_maxlength[i][0] = len(encoder_batch_list[j][i])
            else:
                encoder_questions_list[i].append([])


    for i, (ql, mqlen) in enumerate(zip(encoder_questions_list, encoder_questions_maxlength)):
        for j, q in enumerate(ql):    
            for _ in range(mqlen[0]-len(q)):
                encoder_questions_list[i][j].append(pad_token)


    decoder_questions_list = [[] for i in range(max_questions)]
    decoder_questions_maxlength = [[0] for i in range(max_questions)]

    for i in range(max_questions):
        for j in range(len(batch)):
            if i < len(decoder_batch_list[j]):
                decoder_questions_list[i].append(decoder_batch_list[j][i])
                if decoder_questions_maxlength[i][0] < len(decoder_batch_list[j][i]):
                    decoder_questions_maxlength[i][0] = len(decoder_batch_list[j][i])
            else:
                decoder_questions_list[i].append([])


    for i, (ql, mqlen) in enumerate(zip(decoder_questions_list, decoder_questions_maxlength)):
        for j, q in enumerate(ql):    
            for _ in range(mqlen[0]-len(q)):
                decoder_questions_list[i][j].append(pad_token)



    encoder_batch_matrix = [torch.FloatTensor(enc_q) for enc_q in encoder_questions_list]
    decoder_batch_matrix = [torch.FloatTensor(dec_q) for dec_q in decoder_questions_list]

    return encoder_batch_matrix, decoder_batch_matrix




