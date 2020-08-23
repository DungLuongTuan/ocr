import tensorflow as tf
import numpy as np
import argparse
import pdb
import cv2

from model import Model
from hparams import hparams

def main(args):
    # load pretrain model
    model = Model()
    model.create_model()
    latest = tf.train.latest_checkpoint(hparams.save_path)
    model.checkpoint.restore(latest)

    # inference
    image = cv2.imread(args.image_path)
    result = model.inference(image/255.)
    print(result)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--image_path', help='path to inference image')
    args = parser.parse_args()
    main(args)