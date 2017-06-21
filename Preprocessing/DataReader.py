import json
import h5py

class DataReader():

    def __init__(self, data_path, indicies_path, images_path, images_features_path):

        # save image path in DataReader Object
        self.images_path = images_path

        # read indicies json file with indicies
        self.indicies = json.load(open(indicies_path))
        self.ind2word               = self.indicies['ind2word']
        self.word2ind               = self.indicies['word2ind']
        self.img_metadata           = self.indicies['img_metadata_training']
        self.categories             = self.indicies['categories_training']

        # read data file
        self.data = h5py.File(data_path, 'r')
        self.answers_training           = self.data['answers_training']
        self.game_index_training        = self.data['game_index_training']
        self.image_index_training       = self.data['image_index_training']
        self.image_wh_training          = self.data['image_wh_training']
        self.objects_bbox_training      = self.data['objects_bbox_training']
        self.object_index_training      = self.data['object_index_training']
        self.objects_training           = self.data['objects_training']
        self.question_length_training   = self.data['question_length_training']
        self.questions_training         = self.data['questions_training']
        self.success_training           = self.data['success_training']
        self.correct_object_training    = self.data['correct_object_training']

        # read images_features file
        self.image_features_data = h5py.File(images_features_path, 'r')
        self.all_img_ids        = self.image_features_data['all_img_ids']
        self.all_img_features   = self.image_features_data['all_img_features']



    def get_word2ind(self):
        """  """
        return self.word2ind

    def get_ind2word(self):
        return self.ind2word

    def get_categories_length(self):
        return len(self.categories)

    def get_cat2id(self):
        return self.categories

    def get_target_object(self, game_id):
        return self.correct_object_training[game_id]

    def get_game_ids(self):
        """ returns all game ids """
        game_ids = list()
        for k, v in enumerate(self.game_index_training):
            game_ids.append(k)

        return game_ids


    def get_image_path(self, game_id):
        """ given an game id, returns the image filename """
        data_id = self.game_index_training[game_id]
        img_filename = self.img_metadata[str(data_id)]['filename']
        img_path = self.images_path + '/' + img_filename

        return img_path


    def get_image_id(self, game_id):
        """ given a game id, returns the image id of the image in the game """
        return self.image_index_training[game_id]


    def get_image_url(self, game_id):
        """ given an game id, returns the image url """
        data_id = self.game_index_training[game_id]
        img_url = self.img_metadata[str(data_id)]['coco_url']

        return img_url


    def get_image_width_height(self, game_id):
        """ given a game id, returns the width and height of the image """
        width, height = self.image_wh_training[game_id]
        return width, height


    def get_image_features(self, game_id):
        """ given a game id, return the features of the image for that game """
        image_id = self.get_image_id(game_id)
        feature_id = list(self.all_img_ids).index(image_id)
        return self.all_img_features[feature_id]


    def get_questions(self, game_id):
        """ given game id, returns all questions as list of strings"""
        q_len = self.question_length_training[game_id]
        questions_word_ids = self.questions_training[game_id]

        questions = list()
        for l, qwis in zip(q_len, questions_word_ids):
            if l > 0:
                question = str()
                for i in range(l):
                    question += ' ' + self.ind2word[str(qwis[i])]

                questions.append(question[1:])


        return questions

    def get_question_max_length(self):
        return len(self.questions_training[0][0])


    def get_answers(self, game_id):
        """  given a game id, returns all answers """
        answer_word_ids = self.answers_training[game_id]
        answer_word_ids = [awi for awi in answer_word_ids if awi != 0]
        answers = list()
        for awi in answer_word_ids:
            answers.append(self.ind2word[str(awi)])

        return answers


    def get_object_ids(self, game_id):
        """ given a game id, returns the proposed object ids in the image """
        objects = self.object_index_training[game_id]
        objects = [o for o in objects if o != 0]

        return objects


    def get_object_bbox(self, game_id):
        """ given a game id, returns the bounding boxes of all objects in the same order as get_object_ids """
        bbox = self.objects_bbox_training[game_id]
        bbox = [list(b) for b in bbox if b.all() != 0]

        return bbox

    def get_image_meta(self, game_id):
        """ returns the required metadata for an image, which is the bbox, width and height """
        bbox = get_object_bbox(game_id)
        width, height = get_image_width_height(game_id)
        return [bbox, width, height]

    def get_category_id(self, game_id):
        """ given a game id, returns the categories of the objects in the image """
        categories = self.objects_training[game_id]
        categories = [c for c in categories if c != 0]

        return categories


    def get_success(self, game_id):
        """ given a game id, returns whether the guesses has been successful or not """
        return self.success_training[game_id]


# dr = DataReader(data_path='preprocessed_new.h5', indicies_path='indices_new.json', images_path='val2014')

""" Example of how to use this class
#game_ids = dr.get_game_ids()
#print("game ids", game_ids[:10])
#gid = game_ids[0]
gid = 3
print("img filename", dr.get_image_filename(gid))
print("img url", dr.get_image_url(gid))
print("img w h", dr.get_image_width_height(gid))
print("questions", dr.get_questions(gid))
print("answers", dr.get_answers(gid))
print("object ids", dr.get_object_ids(gid))
print("object bboxes", dr.get_object_bbox(gid))
print("categoires", dr.get_category_id(gid))
print("success", dr.get_success(gid)) """
