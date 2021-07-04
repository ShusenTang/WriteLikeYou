import os
import sys
import json
import random
import numpy as np
from tqdm import tqdm

sys.path.append("../")
from data_utils import random_scale, augment_strokes, format3_zero_pad_to_max_len

class CHN_Reco_DataLoader(object):
    """dataloader for character recognizer (3755-class), format-3"""
    def __init__(self,
                 dirs,
                 is_traing=True,
                 batch_size=1024,
                 max_seq_len=100, 
                 scale_factor=200.0,  # for whole data set
                 random_scale_factor=0.0,  # only for training
                 augment_stroke_prob=0.0  # only for training
                 ):
        self.dirs = dirs
        print("There are %d dirs:" % len(dirs))
        print(" ".join([d.split("/")[-1] for d in dirs]))
        self.batch_size = batch_size  # minibatch size
        self.max_seq_len = max_seq_len
        self.scale_factor = scale_factor  # divide offsets by this factor
        self.is_traing = is_traing
        if not is_traing:
            assert random_scale_factor == 0
            assert augment_stroke_prob == 0
        self.random_scale_factor = random_scale_factor  # data augmentation method
        self.augment_stroke_prob = augment_stroke_prob  # data augmentation method

        self.npy_paths = self._get_valid_npy_path()
        self.size = len(self.npy_paths)
        print("There are %d samples in total." % self.size)

        self.idxes = list(range(self.size))

    def _get_valid_npy_path(self):
        npy_paths = []
        for npy_dir in self.dirs:
            npys = os.listdir(npy_dir)
            for npy in npys:
                if int(npy.split("GB")[-1][:-4]) <= 3755:
                    npy_paths.append(os.path.join(npy_dir, npy))
        return npy_paths

    def shuffle_idxes(self):
        print("Shuffle the idexes!")
        random.shuffle(self.idxes)

    def yield_data(self):
        if self.is_traing:
            self.shuffle_idxes()
        
        for idx in self.idxes:
            npy_path = self.npy_paths[idx]
            zi, zi_len, label = self._get_sample_by_from_npy(npy_path)
            zi_array = format3_zero_pad_to_max_len([zi], self.max_seq_len)[0]  # shape: (max_seq_len, 3)

            yield zi_array, zi_len, label

    def yield_batch_data(self, batch_size=None):
        if self.is_traing:
            self.shuffle_idxes()
        if batch_size is None: batch_size = self.batch_size        
        bn = self.size // batch_size
        for b in range(bn):
            batch_zi = []
            batch_len = []
            batch_label = []
            for idx in self.idxes[batch_size*b: batch_size*b + batch_size]:
                npy_path = self.npy_paths[idx]
                zi, zi_len, label = self._get_sample_by_from_npy(npy_path)
                
                batch_zi.append(zi)
                batch_len.append(zi_len)
                batch_label.append(label)

            batch_zi_array = format3_zero_pad_to_max_len(batch_zi, self.max_seq_len)  # shape: (batch, max_seq_len, 3)
            batch_len_array = np.array(batch_len, dtype=int)  # shape: (batch, )
            batch_label_array = np.array(batch_label, dtype=int)  # shape: (batch, )

            yield batch_zi_array, batch_len_array, batch_label_array


    def _get_sample_by_from_npy(self, npy_path):
        """
        Load an sample (format-3) from npy file (e.g., GB1234.npy).
        """
        label = int(npy_path.split("GB")[-1][:-4]) - 1 # start with 1, so minus 1.
        zi = np.load(npy_path)
        zi_len = len(zi)
        if zi_len > self.max_seq_len:
            zi_len = self.max_seq_len
            zi = zi[:self.max_seq_len]

        zi[:, 0:2] /= self.scale_factor # scale

        # augmentation
        if self.random_scale_factor > 0.0:
            zi = random_scale(zi, self.random_scale_factor)
        if self.augment_stroke_prob > 0.0:
            zi = augment_strokes(zi, self.augment_stroke_prob)
    
        return zi, zi_len, label
