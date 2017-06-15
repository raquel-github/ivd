# Oracle for GuessWhat 

import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

# from torchtext.vocab import load_word_vectors

import numpy
import h5py

def main():
    # Load data file
    file = h5py.File('../Preprocessing/Data/preprocessed.h5', 'r')
    print (file['questions_training'])

    file.close()




if __name__ == '__main__':
    main()
