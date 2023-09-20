# Author: Sung-Wook Park
# Date: 20 Jun 2022
# Last updated: 18 Sep 2023
# --- Ad hoc ---

import argparse
import glob
import os

from PIL import Image
from tqdm import tqdm

def main(config):
    files = glob.glob(config.dataset+'*')

    for f in tqdm(files):
        title, ext = os.path.splitext(f)
        if ext in ['.bmp', '.gif', '.jpg', '.jpeg', '.jpeg 2000', '.png', '.tif', '.tiff', '.webp']:
            img = Image.open(f)
            img_resize = img.resize((config.size, config.size))
            
            isExist = os.path.exists(config.out_dir)
            if not isExist:
                print('Create a new directory because it does not exist')
                os.makedirs(config.out_dir)
            img_resize.save(config.out_dir+os.path.basename(title)+'.jpg')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Free Image Resizer')

    parser.add_argument('--dataset', type=str, required=True, help='Dataset Path')
    parser.add_argument('--size', type=int, default='28', help='Choose Image Size')
    parser.add_argument('--out_dir', type=str, required=True, help='Modified Dataset Path')

    config = parser.parse_args()
    main(config)
