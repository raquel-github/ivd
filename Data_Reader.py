import json
import h5py

class Data_Reader():

    def __init__(self, data_path, indicies_path):

        # read indicies json file
        self.indicies = json.load(open(indicies_path))

        # read data file
        self.data = h5py.File(data_path, 'r')
        self.game_index_training = self.data['game_index_training']
        self.question_length_training = self.data['question_length_training']
        self.questions_training = self.data['questions_training']


    def get_ids(self):
        """ returns all game ids """
        ids = set()

        for k, v in self.game_index_training.items():
            ids.add(k)

        return ids

    def get_image(self, id):
        pass

    def get_question(self, id):
        """ return all questions of the given game id """
        q_len = self.question_length_training[id]
        questions = set()

        pass
