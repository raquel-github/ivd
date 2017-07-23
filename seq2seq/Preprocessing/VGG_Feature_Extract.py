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

    def __init__(self, images_path):

        self.images_path = images_path

        # compute the mean and std. of all pixels for image normalization
        # mean, std = self.get_mean_std()

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

    def get_mean_std(self):
        # get the mean channel value and std of all images
        r_mean, g_mean, b_mean = list(), list(), list()
        r_std, g_std, b_std = list(), list(), list()

        for f in os.listdir(self.images_path+"/"):
            if f.endswith(".jpg"):
                img_p = self.images_path + "/" + f
                channels = numpy.array(Image.open(img_p), dtype=numpy.float) / 255
                # we use imageio as it turns out to be faster
                # channels = imageio.imread(img_p) / 255

                if len(channels.shape) == 2:
                    # bw image, convert to RGB
                    x = numpy.zeros((channels.shape[0], channels.shape[1], 3))
                    x[:,:,0] = channels
                    x[:,:,1] = channels
                    x[:,:,2] = channels
                    channels = x


                # get mean channel value
                r_mean.append(numpy.mean(channels[:,:,0]))
                g_mean.append(numpy.mean(channels[:,:,1]))
                b_mean.append(numpy.mean(channels[:,:,2]))
                # get std channel value
                r_std.append(numpy.std(channels[:,:,0]))
                g_std.append(numpy.std(channels[:,:,1]))
                b_std.append(numpy.std(channels[:,:,2]))

        return [numpy.mean(r_mean), numpy.mean(g_mean), numpy.mean(b_mean)], \
               [numpy.mean(r_std), numpy.mean(g_std), numpy.mean(b_std)]


    def get_features(self, img_p):
        """ Given an image path, this function will return the VGG features """
        img = Image.open(img_p)
        if len(numpy.array(img).shape) == 2:
            rgb = Image.new('RGB',img.size)
            rgb.paste(img)
            img = rgb
            print(img_p)
            print(numpy.array(img).shape)
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
