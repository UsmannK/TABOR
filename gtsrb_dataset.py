"""
GTSRB Dataset
"""
import os
from math import ceil
import glob
import random
import pandas as pd
import numpy as np
from tqdm import trange
from PIL import Image

class GTSRBDataset:
    """
    GTSRB data loader. This is generally ridiculous but since the dataset is so
    small (120MB all told), we can just load the whole thing into memory!
    """

    def __init__(self, val_split=0.2, data_dir='GTSRB'):
        self.val_split = val_split
        self.data_dir = data_dir
        csv_files = glob.glob('{}/Final_Training/Images/*/*.csv'.format(
            data_dir
        ))
        self.process_csvs(csv_files)
        self.load_imgs()

        print("Processed {} annotations".format(self.num_total))
        print("{} Train examples".format(self.num_train))
        print("{} Test examples".format(self.num_test))
        print("{}/{} = {:0.2f}".format(self.num_test, self.num_total,
         self.num_test/self.num_total))

    def process_csvs(self, csv_files):
        """
        Extract information from scattered annotation files
        """
        self.train_img_fnames = []
        self.test_img_fnames = []
        self.train_labels = []
        self.test_labels = []

        for annotation_file in csv_files:
            annotation = pd.read_csv(annotation_file, delimiter=';')
            # Image filenames are stored as {sign_id}_{photo_num}.ppm
            # Images that share a sign_id are the same physical sign
            # Make sure to not leak the same sign in the train/val split
            img_fnames = annotation['Filename']
            cls_id = annotation['ClassId'][0]
            # Get unique sign ids, shuffle them explicitly, and cut off
            # ceil(val_split*len) of them
            sign_ids = set([fname.split('_')[0] for fname in img_fnames])
            sign_ids = list(sign_ids)
            random.shuffle(sign_ids)
            split_id = ceil(self.val_split * len(sign_ids))

            train_sign_ids = set(sign_ids[split_id:])
            test_sign_ids = set(sign_ids[:split_id])

            for img_fname in img_fnames:
                sign_id = img_fname.split('_')[0]
                if sign_id in train_sign_ids:
                    self.train_img_fnames.append(img_fname)
                    self.train_labels.append(cls_id)
                elif sign_id in test_sign_ids:
                    self.test_img_fnames.append(img_fname)
                    self.test_labels.append(cls_id)
                else:
                    raise KeyError(sign_id)
        
        self.num_train = len(self.train_img_fnames)
        self.num_test = len(self.test_img_fnames)
        self.num_total = self.num_train + self.num_test
    
    def load_imgs(self):
        """
        Load image data itself into numpy arrays
        """
        self.train_images = np.empty((self.num_train, 32, 32, 3), dtype=np.uint8)
        self.test_images = np.empty((self.num_test, 32, 32, 3), dtype=np.uint8)
        self.train_labels = np.array(self.train_labels, dtype=np.uint8)
        self.test_labels = np.array(self.test_labels, dtype=np.uint8)

        image_base_path = '{}/Final_Training/Images/'.format(self.data_dir)

        for idx in trange(self.num_train, desc='Load train images', ncols=80):
            cls_id = self.train_labels[idx]
            fname = self.train_img_fnames[idx]
            img_path = os.path.join(image_base_path, '{:05d}'.format(cls_id), fname)
            img = np.array(Image.open(img_path).resize((32, 32)))
            self.train_images[idx] = img

        for idx in trange(self.num_test, desc='Load test images', ncols=80):
            cls_id = self.test_labels[idx]
            fname = self.test_img_fnames[idx]
            img_path = os.path.join(image_base_path, '{:05d}'.format(cls_id), fname)
            img = np.array(Image.open(img_path).resize((32, 32)))
            self.test_images[idx] = img

def gtsrb_signname(classid):
    labels = {
    0 : "speed limit 20 (prohibitory)",
    1 : "speed limit 30 (prohibitory)",
    2 : "speed limit 50 (prohibitory)",
    3 : "speed limit 60 (prohibitory)",
    4 : "speed limit 70 (prohibitory)",
    5 : "speed limit 80 (prohibitory)",
    6 : "restriction ends 80 (other)",
    7 : "speed limit 100 (prohibitory)",
    8 : "speed limit 120 (prohibitory)",
    9 : "no overtaking (prohibitory)",
    10 : "no overtaking (trucks) (prohibitory)",
    11 : "priority at next intersection (danger)",
    12 : "priority road (other)",
    13 : "give way (other)",
    14 : "stop (other)",
    15 : "no traffic both ways (prohibitory)",
    16 : "no trucks (prohibitory)",
    17 : "no entry (other)",
    18 : "danger (danger)",
    19 : "bend left (danger)",
    20 : "bend right (danger)",
    21 : "bend (danger)",
    22 : "uneven road (danger)",
    23 : "slippery road (danger)",
    24 : "road narrows (danger)",
    25 : "construction (danger)",
    26 : "traffic signal (danger)",
    27 : "pedestrian crossing (danger)",
    28 : "school crossing (danger)",
    29 : "cycles crossing (danger)",
    30 : "snow (danger)",
    31 : "animals (danger)",
    32 : "restriction ends (other)",
    33 : "go right (mandatory)",
    34 : "go left (mandatory)",
    35 : "go straight (mandatory)",
    36 : "go right or straight (mandatory)",
    37 : "go left or straight (mandatory)",
    38 : "keep right (mandatory)",
    39 : "keep left (mandatory)",
    40 : "roundabout (mandatory)",
    41 : "restriction ends (overtaking) (other)",
    42 : "restriction ends (overtaking (trucks)) (other)"
    }
    return labels[classid]

if __name__ == '__main__':
    # for profiling time to load
    # currently ~3.5s on my laptop
    _ = GTSRBDataset()
