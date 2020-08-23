from augment_filters import *
import numpy as np
import subprocess
import argparse
import random
import pdb
import cv2
import os

def main(args):
    data = []
    with open(os.path.join(args.inp_path, "label.txt"), "r") as f:
        for row in f:
            filename, label = row[:-1].split("\t")
            data.append((filename, label))

    w_data = []
    for i, (filename, label) in enumerate(data):
        print(i, filename)
        image = cv2.imread(os.path.join(args.inp_path, filename))
        for filtername in filters.keys():
            img = filters[filtername](image.copy())
            w_data.append((filename[:-4] + '_' + filtername + '.jpg', label))
            cv2.imwrite(os.path.join(args.out_path, filename[:-4] + '_' + filtername + '.jpg'), img)
        w_data.append((filename, label))
        cv2.imwrite(os.path.join(args.out_path, filename), image)

    random.shuffle(w_data)
    with open(os.path.join(args.out_path, 'label.txt'), 'w') as f:
        for filename, label in w_data:
            f.write(filename + '\t' + label + '\n')
    # subprocess.call(["cp " + os.path.join(args.inp_path, "label.txt") + " " + args.out_path], shell = True)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--inp_path", default = "data/test/transform/raw")
    parser.add_argument("--out_path", default = "data/test/transform/test")
    args = parser.parse_args()
    main(args)

"""
    python3 augment_data.py --inp_path="/home/dunglt/cmnd/po/data/train/" --out_path="/home/dunglt/cmnd/dung/data/random_ns_v2/train"
"""
