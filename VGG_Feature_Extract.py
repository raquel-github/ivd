import vgg
import io
import requests
from PIL import Image
import imageio
from torchvision import models, transforms
from torch.autograd import Variable
import torch.nn as nn
import numpy

class VGG_Feature_Extract():

    def __init__(self, image_paths):

        # compute the mean and std. of all pixels for image normalization
        mean, std = self.get_mean_std(image_paths)

        # noraliztion pipeline: (img-mean) / std
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

    def get_mean_std(self, image_paths):
        # get the mean channel value and std of all images
        r_mean, g_mean, b_mean = list(), list(), list()
        r_std, g_std, b_std = list(), list(), list()
        for img_p in image_paths:
            #channels = numpy.array(Image.open(img_p), dtype=numpy.float) / 255
            # we use imageio as it turns out to be faster
            channels = imageio.imread(img_p) / 255

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
        img_tensor = self.preprocess(Image.open(img_p))
        img_tensor.unsqueeze_(0)
        img_variable = Variable(img_tensor)
        return self.model(img_variable)

"""
Example How To Extract Features
image_paths = list()
image_paths.append('val2014/COCO_val2014_000000000042.jpg')
image_paths.append('val2014/COCO_val2014_000000000073.jpg')
image_paths.append('val2014/COCO_val2014_000000000074.jpg')
image_paths.append('val2014/COCO_val2014_000000000133.jpg')
image_paths.append('val2014/COCO_val2014_000000000136.jpg')


fe = VGG_Feature_Extract(image_paths)

for img_p in image_paths:
    fe.get_features(img_p)
"""
