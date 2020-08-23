from model import Model

import tensorflow as tf
import numpy as np
import pdb
import os

def main():
    ### create model
    model = Model()
    model.create_model()
    ### train new model
    model.train()

if __name__ == '__main__':
    main()