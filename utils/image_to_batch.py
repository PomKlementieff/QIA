# Author: Sung-Wook Park
# Date: 20 Jun 2022
# Last updated: 18 Sep 2023
# --- Ad hoc ---

import argparse
import numpy as np
import random

from PIL import Image
from tqdm import tqdm

def train_test_shuffle(x_train, y_train, x_test, y_test):

    train = [[x, y] for x, y in zip(x_train, y_train)]
    random.shuffle(train)
    x_train = [n[0] for n in train]
    y_train = [n[1] for n in train]

    test = [[x, y] for x, y in zip(x_test, y_test)]
    random.shuffle(test)
    x_test = [n[0] for n in test]
    y_test = [n[1] for n in test]

    return x_train, y_train, x_test, y_test

def main(config):
    train_img_list = []
    train_label_list = []

    with open('trainval.txt', 'r') as f:
        while True:
            line = f.readline()
            if not line: break
            
            train_img_list.append(config.dataset+(line.rstrip('\n').split(' '))[0]+'.jpg')
            train_label_list.append((line.rstrip('\n').split(' '))[-1])

    x_train = np.zeros((len(train_img_list), config.size, config.size), dtype='float32')

    for idx, path in tqdm(enumerate(train_img_list[:])):
        image = np.array(Image.open(path))
        x_train[idx, :, :] = image
        
    test_img_list = []
    test_label_list = []

    with open('test.txt', 'r') as f:
        while True:
            line = f.readline()
            if not line: break            
            test_img_list.append(config.dataset+(line.rstrip('\n').split(' '))[0]+'.jpg')
            test_label_list.append((line.rstrip('\n').split(' '))[-1])
            
    x_test = np.zeros((len(test_img_list), config.size, config.size), dtype='float32')
    
    for idx, path in tqdm(enumerate(test_img_list[:])):
        image = np.array(Image.open(path))
        x_test[idx, :, :] = image

    x_train, y_train, x_test, y_test = train_test_shuffle(x_train, train_label_list, x_test, test_label_list) 
    x_train = np.array(x_train, dtype='float32')
    y_train = np.array(y_train, dtype='uint8')
    x_test = np.array(x_test, dtype='float32')
    y_test = np.array(y_test, dtype='uint8')

    xy = (x_train, y_train), (x_test, y_test)
    np.save('clamm.npy', xy)
    
    (x_train, y_train), (x_test, y_test) = np.load('clamm.npy', allow_pickle= True)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Image to Batch')
    
    parser.add_argument('--dataset', type=str, required=True, help='Dataset Path')
    parser.add_argument('--size', type=int, default='28', help='Choose Image Size')

    config = parser.parse_args()
    main(config)
