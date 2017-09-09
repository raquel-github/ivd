import torch
import torch.nn as nn
from torchvision import models, transforms
from torch.autograd import Variable
from torchvision.models import resnet

import numpy as np
from PIL import Image

use_cuda = torch.cuda.is_available()

class ResNet_Feature_Extract():
    """docstring for ResNet_Feature_Extract"""
    def __init__(self):
        super(ResNet_Feature_Extract, self).__init__()

        self.model = resnet.resnet152(pretrained=True)

        self.model.fc = nn.Linear(512 * 4, 512 * 4)

        # Using the imagenet mean and std
        mean =  [0.485, 0.456, 0.406]
        std  =  [0.229, 0.224, 0.225]

        # normaliztion pipeline: (img-mean) / std
        self.normalize = transforms.Normalize(
            mean=mean,
            std=std
        )

        # preprocessing pipeline
        self.preprocess = transforms.Compose([
            transforms.Scale(224),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            self.normalize
         ])

        if use_cuda:
            self.model.cuda()

    def get_features(self, img_p):
        """ Given an image path, this function will return the VGG features """
        img = Image.open(img_p)
        if len(np.array(img).shape) == 2:
            rgb = Image.new('RGB',img.size)
            rgb.paste(img)
            img = rgb
            
        img_tensor = self.preprocess(img)
        img_tensor.unsqueeze_(0)
        if use_cuda:
            img_variable = Variable(img_tensor).cuda()
        else:
            img_variable = Variable(img_tensor)
        # print(img_p)
        return self.model(img_variable)
