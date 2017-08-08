import vgg
import io
import requests
from PIL import Image
import torch
from torchvision import models, transforms
from torch.autograd import Variable
import torch.nn as nn
import numpy
import os

use_cuda = torch.cuda.is_available()

class VGG_Feature_Extract():

    def __init__(self):
        # load preptrained model
        self.model = vgg.vgg16_bn(pretrained=True)


        # replace fullyconnceted layer to get onyl features (and not classification)
        self.model.classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(4096, 4096)
        )

        if use_cuda:
            self.model.cuda();



    def get_features(self, img_p):
        """ Given an image path, this function will return the VGG features """
        img = Image.open(img_p)
        if len(numpy.array(img).shape) == 2:
            rgb = Image.new('RGB',img.size)
            rgb.paste(img)
            img = rgb
            # print(img_p)
            # print(numpy.array(img).shape)
        img_tensor = self.preprocess(img)
        img_tensor.unsqueeze_(0)
        if use_cuda:
            img_variable = Variable(img_tensor).cuda()
        else:
            img_variable = Variable(img_tensor)
        # print(img_p)
        return self.model(img_variable)


# Example How To Extract Features
# images_path = str('../data/Sample/MS_COCO')


# fe = VGG_Feature_Extract(images_path)

# for f in os.listdir(images_path+"/"):
#             if f.endswith(".jpg"):
#                 img_p = images_path + "/" + f
#                 print(fe.get_features(img_p))
