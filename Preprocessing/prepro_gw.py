# Preprocessing data for the GuessWhat?! task
# 
# Code based on earlier work at https://github.com/batra-mlp-lab/visdial/blob/master/data/prepro.py

# Imports
import os
import json
import h5py
import argparse
import numpy as np
from nltk.tokenize import word_tokenize

# Function to tokenize the data. 
# Param     Raw JSON data
# Return    Dictionary with id, length of q/a's, objects for all games
# Return    The tokens for each question
# Return    The tokens for each answer
# Return    Dictionary that maps category IDs to category names
# Return    The word count for each word in both the questions and answers
# Return    The maximum number of question/answer pairs for a game
# Return    The maximum number of objects for a game
def tokenize_data(data):

    # Initialize all lists, dictionaries and counters
    res, word_counts, categories = {}, {}, {}
    question_tokens, answer_tokens = [], []
    max_conv_length = 0
    max_objects = 0
    print ("Tokenizing questions and answers")

    # Loop over all games
    for game in data:

        # Fetch the initial data
        res[game['id']] = {'id': game['id'], 'length': len(game['qas'])}

        # Get the length of the conversation and the number of objects
        max_conv_length = max(max_conv_length, len(game['qas']))
        max_objects = max(max_objects, len(game['objects']))

        # Save the q/a IDs and objects for this game
        qa_ids = []
        objects = []

        # For each object, save the category ID and bounding box
        for obj in game['objects']:
            item = {}
            item['category'] = obj['category_id']
            item['bounding'] = obj['bbox']
            item['id'] = obj['id']
            objects.append(item)

            # Fetch the category ID
            categories[obj['category_id']] = obj['category']

        # For each Q/A pair, save the question and answer
        for qa in game['qas']:
            question = qa['question']
            answer = qa['answer']

            # Save the ID of this Q/A pair, for it to be associated with the game
            qa_ids.append(len(question_tokens))

            # Append the tokenized question and answer (only one word) to the list of tokens
            tokenized_question = word_tokenize(question)
            question_tokens.append(tokenized_question)
            answer_tokens.append(answer)

            # Update the word count for every word in the question and answer
            for word in tokenized_question:
                word_counts[word] = word_counts.get(word, 0) + 1
            word_counts[answer] = word_counts.get(answer, 0) + 1 

        # Save metadata about the game
        res[game['id']]['qa_ids'] = qa_ids
        res[game['id']]['objects'] = objects
        res[game['id']]['num_objects'] = len(game['objects'])
        res[game['id']]['success'] = 1 if game['status'] == 'success' else 0

        # Save data about the image
        res[game['id']]['image'] = {}
        res[game['id']]['image']['coco_url']  = game['image']['coco_url']
        res[game['id']]['image']['filename'] = game['image']['file_name']
        res[game['id']]['image']['width'] = game['image']['width']
        res[game['id']]['image']['height'] = game['image']['height']

    # Return all gathered data
    return res, question_tokens, answer_tokens, categories, word_counts, max_conv_length, max_objects

# Function to encode questions and answers according to the vocabulary
# Param     The tokens for each question
# Param     The tokens for each answer
# Param     The dictionary that maps all words to indices in the vocabulary
# Return    The indices for each question
# Return    The indices for each answer
# Return    The maximum question length
def encode_vocab(question_tokens, answer_tokens, word2ind):

    # Save the indices of the questions and answers
    question_indices = []
    answer_indices = []

    # Store the maximum question length
    max_question_length = 0

    # For each question, fetch the index of the token or return the default token ('UNK') index
    for i in question_tokens:
        question = [word2ind.get(word, word2ind['UNK']) for word in i]
        question_indices.append(question)
        max_question_length = max(max_question_length, len(question))

    # For each answer, store the 'Yes'/'No'/'N/A' token index
    for i in answer_tokens:
        answer = word2ind.get(i, word2ind['UNK'])
        answer_indices.append(answer)

    # Return all data
    return question_indices, answer_indices, max_question_length

# Convert all data to data matrices
# Param     Dictionary with id, length of q/a's, objects for all games
# Param     The indices for each question
# Param     The indices for each answer
# Param     The maximum question length
# Param     The maximum converstation length (number of Q/A pairs)
# Param     The maximum number of objects in a game
# Return    Data matrix containing all questions
# Return    Data matrix containing all question lenghts
# Return    Data matrix containing all answers
# Return    Data matrix containing all image indices (mapped from i => i)
# Return    Data matrix containing info about all objects
# Return    Data matrix containing info about object bounding boxes
# Return    Data matrix mapping object X in a game to the actual object ID
# Return    Data matrix mapping number of game to game ID
# Return    Data matrix containing width and height of images
# Return    Data matrix containing filename and coco_url of images
# Return    Data matrix containing wheter a game was succesful or not
def create_data_matrices(data, question_indices, answer_indices, max_question_length, max_conv, max_objects):

    # Fetch the number of items (games)
    num_items = len(data.keys())

    # Initialize data matrices with zeros
    questions = np.zeros([num_items, max_conv, max_question_length])
    answers = np.zeros([num_items, max_conv])
    objects = np.zeros([num_items, max_objects])
    objects_bbox = np.zeros([num_items, max_objects, 4])
    image_index = np.zeros(num_items)
    object_index = np.zeros([num_items, max_objects])
    success = np.zeros(num_items)
    game_index = np.zeros(num_items)
    question_length = np.zeros([num_items, max_conv], dtype=np.int)
    image_wh = np.zeros([num_items, 2])
    image_meta = {}

    # Loop over all games
    for i in range(num_items):

        # Get the ID of the image (and the game) and save it
        img_id = data.keys()[i]
        image_index[i] = i 

        # Save if the game was succesful or not
        success[i] = data[img_id]['success']

        # Store the game/image ID
        game_index[i] = img_id

        # Store image width en height
        image_wh[i][0] = data[img_id]['image']['width']
        image_wh[i][1] = data[img_id]['image']['height']

        # Store image file name and coco_url
        image_meta[img_id] = {}
        image_meta[img_id]['filename'] = data[img_id]['image']['filename']
        image_meta[img_id]['coco_url'] = data[img_id]['image']['coco_url']

        # Loop over all Q/A pairs in this game
        for j in range(data[img_id]['length']):

            # Fetch one of the questions and answers that belong to this game
            question = question_indices[data[img_id]['qa_ids'][j]]
            answer = answer_indices[data[img_id]['qa_ids'][j]]

            # Add info about this Q/A pair to the data matrices
            question_length[i][j] = len(question)
            questions[i][j][0:question_length[i][j]] = question
            answers[i][j] = answer

        # Loop over all objects in this image
        for j in range(data[img_id]['num_objects']):

            # Store info about this object in the data matrices
            obj = data[img_id]['objects'][j]
            objects[i][j] = obj['category']
            objects_bbox[i][j] = obj['bounding']
            object_index[i][j] = obj['id']

    # Return all data matrices
    return questions, question_length, answers, image_index, game_index, objects, objects_bbox, object_index, image_wh, image_meta, success



if __name__ == '__main__':
    # =======================================================
    #                   Reading the JSON data
    # =======================================================
    print("Reading JSON data...")

    # Open the file 
    with open('Data/guesswhat.small.test.jsonl') as f:
        content = f.readlines()

    # Since the data is in JSONL format instead of JSON, the entire file is not 
    # a valid JSON file. However, each line is valid JSON, so we have to parse
    # every line as a separate JSON entry.
    data_training = [json.loads(x.strip()) for x in content]
    
    # Tokenize the data
    data_training, question_training_tokens, answer_training_tokens, categories_training, word_counts_training, max_conv_training, max_objects_training = tokenize_data(data_training)

    # =======================================================
    #                 Building the vocabulary
    # =======================================================
    print("Building vocabulary")

    # Set the minimum number of occurences of a word for it to be included in the vocab
    word_counts_training['UNK'] = 5

    # Include all words that need to be in the vocabulary
    vocab = [word for word in word_counts_training if word_counts_training[word] >= 5]
    
    # Print the number of (unique) words
    print("Number of words in vocabulary: %d" % len(vocab))

    # Create a word -> index dictionary for all words
    word2ind = {word:word_ind for word_ind, word in enumerate(vocab)}

    # Create a index -> word dictionary for all indices
    ind2word = {word_ind:word for word, word_ind in word2ind.items()}

    # =======================================================
    #                Encode based on vocab
    # =======================================================
    print("Encode based on vocabulary")
    question_training_indices, answer_training_indices, max_question_length = encode_vocab(question_training_tokens, answer_training_tokens, word2ind)

    # =======================================================
    #                  Create data matrices
    # =======================================================
    print("Create data matrices")
    questions_training, question_length_training, answers_training, image_index_training, game_index_training, objects_training, objects_bbox_training, object_index_training, image_wh_training, image_meta_training, success_training = create_data_matrices(data_training, question_training_indices, answer_training_indices, max_question_length, max_conv_training, max_objects_training)

    # =======================================================
    #                     Save data
    # =======================================================
    print("Save HDF5 file")
    file = h5py.File('Data/preprocessed.h5', 'w')

    print(image_meta_training)

    file.create_dataset('questions_training', dtype='uint32', data=questions_training)
    file.create_dataset('question_length_training', dtype='uint32', data=question_length_training)
    file.create_dataset('answers_training', dtype='uint32', data=answers_training)
    # file.create_dataset('image_index_training', dtype='uint32', data=image_index_training)
    file.create_dataset('game_index_training', dtype='uint32', data=game_index_training)
    file.create_dataset('objects_training', dtype='uint32', data=objects_training)
    file.create_dataset('objects_bbox_training', dtype='float32', data=objects_bbox_training)
    file.create_dataset('success_training', dtype='uint32', data=success_training)
    file.create_dataset('object_index_training', dtype='uint32', data=object_index_training)
    file.create_dataset('image_wh_training', dtype='uint32', data=image_wh_training)

    file.close()

    # Export word2ind and ind2word to JSON format
    print("Save JSON file")
    
    out = {}
    out['ind2word'] = ind2word
    out['word2ind'] = word2ind
    out['categories'] = categories_training
    out['img_metadata'] = image_meta_training
    json.dump(out, open('Data/indices.json', 'w'))

    # We're done.
    print("Done.")
