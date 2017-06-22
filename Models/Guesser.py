import torch
import torch.autograd as autograd
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F

class Guesser(nn.Module):

    def __init__(self, hidden_encoder_dim, categories_length, cat2id, object_embedding_dim):
        """
        Parameters
        hidden_encoder_dim          Dimensionality of the hidden state of the encoder
        categories_length           Number of object categories_length
        cat2id                      Dictionary to convert a category into an id
        object_embedding_dim        Dimensionality of the object embeddings
        """
        super(Guesser, self).__init__()


        self.spatial_dim = 8
        self.categories_length = categories_length
        self.object_embedding_dim = object_embedding_dim
        self.hidden_encoder_dim = hidden_encoder_dim
        self.cat2id = cat2id

        self.object_embedding_model = nn.Embedding(self.categories_length, self.object_embedding_dim)

        self.mlp_model = nn.Sequential(
            nn.Linear(self.object_embedding_dim + self.spatial_dim, 64),
            nn.RelU(),
            nn.Linear(64, self.hidden_encoder_dim)
        )


    def get_cat2id(self, cat):
        onehot = torch.zeros(self.categories_length)
        onehot[self.cat2id[cat]] = 1
        return onehot


    def img_spatial(img_meta):
        """ returns the spatial information of a bounding box """
        bboxes          = img_meta[0] # gets all bboxes in the image

        width           = img_meta[1]
        height          = img_meta[2]
        image_center_x  = width / 2
        image_center_y  = height / 2

        spatial = Variable(torch.FloatTensor(len(bboxes), 8))

        for i, bbox in enumerate(bboxes):
            x_min = (min(bbox[0], bbox[2]) - image_center_x) / image_center_x
            y_min = (min(bbox[1], bbox[3]) - image_center_y) / image_center_y
            x_max = (max(bbox[0], bbox[2]) - image_center_x) / image_center_x
            y_max = (max(bbox[1], bbox[3]) - image_center_y) / image_center_y
            x_center = (x_min + x_max) / 2
            y_center = (y_min + y_max) / 2

            w_box = x_max - x_min
            h_box = y_max - y_min

            spatial[i] = torch.FloatTensor([x_min, y_min, x_max, y_max, x_center, y_center, w_box, h_box])


        return spatial


    def forward(self, hidden_encoder, img_meta, object_categories):

        # get the object embeddings of the objects in the image
        object_embeddings = Variable(torch.zeros(len(object_categories), self.object_embedding_dim))
        for i, obj in enumerate(object_categories):
            obj_embeddings[i] = self.object_embedding_model(obj)

        # get the spatial info
        spatial = img_spatial(img_meta)

        mlp_in = Variable(torch.cat([spatial, object_embeddings]))

        proposed_embeddings = mlp_model(mlp_in)

        hidden_encoder = torch.cat([hidden_encoder[0]] * len(object_categories))

        

        return F.log_softmax(torch.mm(proposed_embeddings, hidden_encoder[0].view(self.hidden_encoder_dim, -1)).t())
