import torch
import torch.autograd as autograd
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F

class Guesser(nn.Module):

    def __init__(self, hidden_encoder_dim, categories_length, cat2id):
        super(Guesser, self).__init__()


        self.spatial_dim = 8
        self.categories_length = categories_lenght
        self.object_embedding_dim = 20
        self.hidden_encoder_dim = hidden_encoder_dim
        self.cat2id = cat2id

        self.object_embedding_model = nn.Sequential(
            nn.Linear(self.categories_length, 60)
            nn.ReLU()
            nn.Linear(60, self.object_embedding_dim)
            nn.Tanh()
        )

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
        bbox            = img_meta[0]
        width           = img_meta[1]
        height          = img_meta[2]
        image_center_x  = width / 2
        image_center_y  = height / 2


        x_min = (min(bbox[0], bbox[2]) - image_center_x) / image_center_x
        y_min = (min(bbox[1], bbox[3]) - image_center_y) / image_center_y
        x_max = (max(bbox[0], bbox[2]) - image_center_x) / image_center_x
        y_max = (max(bbox[1], bbox[3]) - image_center_y) / image_center_y
        x_center = (x_min + x_max) / 2
        y_center = (y_min + y_max) / 2

        w_box = x_max - x_min
        h_box = y_max - y_min


        return Variable(torch.FloatTensor([x_min, y_min, x_max, y_max, x_center, y_center, w_box, h_box]))


    def forward(self, hidden_encoder, img_meta, object_categories):


        obj_onehot = Variable(torch.zeros(len(object_categories), self.categories_length))
        for oid, o_cat in enumerate(object_categories):
            obj_onehot[oid] = get_cat2id(o_cat)

        object_embeddings = self.object_embedding_model(obj_onehot)

        spatial = img_spatial(img_meta)
        mlp_in = Variable(torch.cat([spatial, object_embeddings]))

        proposed_embeddings = mlp_model(mlp_in)

        return F.log_softmax(torch.mm(proposed_embeddings, hidden_encoder[0].view(self.hidden_encoder_dim, -1)).t())
