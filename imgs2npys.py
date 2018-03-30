import argparse
import tensorflow as tf
parser = argparse.ArgumentParser()
add_arg = parser.add_argument

add_arg('--input', default='/home/yyl/pjs/pycharm-remote-data/imgs', type=str,     \
        help='Output dir set in \'prepare.py\'.')
add_arg('--output', default='/home/yyl/pjs/pycharm-remote-data/data-512/', type=str,    \
        help='npy\'s will be stored here.')
add_arg('--test-set-size', default=50, type=int, \
        help='Number of images to be reserved for test set.')

args = parser.parse_args()

from glob import glob
from skimage.io import imread
from random import shuffle
from os.path import exists
from os import makedirs
from shutil import rmtree
from PIL import Image
from resizeimage import resizeimage
import matplotlib.pyplot as plt
import numpy as np
if __name__ == '__main__':
    train_path = args.output + 'train.npy'
    test_path  = args.output + 'test.npy'

    # imgs_path = 'imgs/*'

    files = glob(args.input + '/*.png')

    dataset_size = len(files)
    training_set_size = dataset_size - args.test_set_size

    assert (dataset_size > 0)                 , 'Dataset is empty.'
    assert (args.test_set_size > 0)           , 'Invalid test_set_size.'
    assert (args.test_set_size < dataset_size), 'Invalid test_set_size.'

    if not exists(args.output):
        makedirs(args.output)

    shuffle(files)

    train_imgs = np.empty(shape=(training_set_size, 512, 512, 3), dtype=np.uint8)

    print('Generate \'train.npy\' ...')
    for idx, fname in enumerate(files[args.test_set_size:]):
        img = Image.open(fname)
        img = resizeimage.resize_cover(img,[512,512])
        train_imgs[idx] = img
    print('Done.')

    print('Save \'train.npy\' ...')
    np.save(file=train_path , arr=train_imgs , allow_pickle=False)
    print('Done.')

    test_imgs = np.empty(shape=(args.test_set_size, 512, 512, 3), dtype=np.uint8)

    print('Generate \'test.npy\' ...')
    for idx, fname in enumerate(files[:args.test_set_size]):
        #test_imgs[idx] = imread(fname)
        img = Image.open(fname)
        img = resizeimage.resize_cover(img,[512,512])
        test_imgs[idx] = img
    print('Done.')

    print('Save \'test.npy\' ...')
    np.save(file=test_path , arr=test_imgs , allow_pickle=False)
    print('Done.\nRemoving temporary dataset files ...')

    rmtree(args.input)
    print('Done.')

