# Author: Sung-Wook Park
# Date: 20 Jun 2022
# Last updated: 18 Sep 2023
# --- Ad hoc ---

import argparse
import numpy as np

from PIL import Image
from tqdm import tqdm

def read_image(path):
    image = np.array(Image.open(path))
    return image.reshape(image.shape[0], image.shape[1], 1)

def onehot_encode_label(unique_label_list, label_list, index):
    onehot_label = unique_label_list == label_list[index]
    onehot_label = onehot_label.astype(np.uint8)
    return onehot_label

def main(config):
    train_img_list = []
    train_label_list = []

    with open('trainval.txt', 'r') as f:
        while True:
            line = f.readline()
            if not line: break
            
            train_img_list.append(config.dataset+(line.rstrip('\n').split(' '))[0]+'.jpg')
            train_label_list.append((line.rstrip('\n').split(' '))[-1])
            
    unique_label_list = np.unique(train_label_list)
    nb_classes = len(unique_label_list)

    x_train = np.zeros((len(train_img_list), config.size, config.size, config.channel), dtype='float32')
    y_train = np.zeros((len(train_img_list), nb_classes), dtype='uint8')
    
    for idx, path in tqdm(enumerate(train_img_list[:])):
        image = read_image(path)
        onehot_label = onehot_encode_label(unique_label_list, train_label_list, idx)
        x_train[idx, :, :, :] = image
        y_train[idx, :] = onehot_label
        
    test_img_list = []
    test_label_list = []

    with open('test.txt', 'r') as f:
        while True:
            line = f.readline()
            if not line: break            
            test_img_list.append(config.dataset+(line.rstrip('\n').split(' '))[0]+'.jpg')
            test_label_list.append((line.rstrip('\n').split(' '))[-1])
            
    x_test = np.zeros((len(test_img_list), config.size, config.size, config.channel), dtype='float32')
    y_test = np.zeros((len(test_img_list), nb_classes), dtype='uint8')
    
    for idx, path in tqdm(enumerate(test_img_list[:])):
        image = read_image(path)
        onehot_label = onehot_encode_label(unique_label_list, test_label_list, idx)
        x_test[idx, :, :, :] = image
        y_test[idx, :] = onehot_label
    
    xy = (x_train, y_train), (x_test, y_test)
    np.save('clamm.npy', xy)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Image to Batch')
    
    parser.add_argument('--dataset', type=str, required=True, help='Dataset Path')
    parser.add_argument('--size', type=int, default='28', help='Choose Image Size')
    parser.add_argument('--channel', type=int, default='1', help='Choose Image Channel')

    config = parser.parse_args()
    main(config)
