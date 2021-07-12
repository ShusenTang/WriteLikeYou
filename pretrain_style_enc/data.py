import os
import sys
import json
import time
import copy
import pickle
import random
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from multiprocessing import Pool

sys.path.append("../")
from data_utils import random_scale, augment_strokes, format3_zero_pad_to_max_len, draw_one_hanzi



def _get_sample_by_from_npy(npy_path):
    """
    load sample (format3)
    """
    max_seq_len = 100
    random_scale_factor = 0.0
    augment_stroke_prob = 0.0

    style_str = npy_path.split("/")[-2] # XX/123/GBXXXX.npy -> 123
    
    zi = np.load(npy_path)
    zi_len = len(zi)
    if zi_len > max_seq_len:
        zi_len = max_seq_len
        zi = zi[:max_seq_len]

    zi[:, 0:2] /= 200.0 # scale

    if random_scale_factor > 0.0:
        zi = random_scale(zi, random_scale_factor)
    if augment_stroke_prob > 0.0:
        zi = augment_strokes(zi, augment_stroke_prob)

    return zi, zi_len

class CHN_Style_DataLoader(object):
    """dataloader for style classifier"""
    def __init__(self,
                 npz_files, 
                 GB_range = (1, 3755), 
                 is_traing=True,
                 num_writer_per_batch=64,
                 num_entry_per_writer=16,
                 max_seq_len=110, 
                 scale_factor=200.0,  # for whole data set
                 random_scale_factor=0.0,  # only for training
                 augment_stroke_prob=0.0  # only for training
                 ):
        self.npz_files = npz_files
        self.total_writer_num = len(npz_files)

        
        print("%d npz files, namely %d writers:" % (self.total_writer_num, self.total_writer_num))
        print(" ".join([d.split("/")[-1] for d in npz_files]))
        self.GB_range = GB_range
        print("GB range:", GB_range)
        self.num_writer_per_batch = num_writer_per_batch
        self.num_entry_per_writer = num_entry_per_writer
        self.batch_size = num_writer_per_batch * num_entry_per_writer  # minibatch size
        self.max_seq_len = max_seq_len
        self.scale_factor = scale_factor  # divide offsets by this factor
        self.is_traing = is_traing
        if not is_traing:
            assert random_scale_factor == 0
            assert augment_stroke_prob == 0
        self.random_scale_factor = random_scale_factor  # data augmentation method
        self.augment_stroke_prob = augment_stroke_prob  # data augmentation method

        self._get_all_valid_data()

    def _get_all_valid_data(self):
        """remove the samples outside GB_range
        arr_list_list: list of list, arr_list_list[i] contains all npy in npz_files[i] which inside GB_range.
        """
        self.arr_list_list = []
        self.size = 0
        print("Reading all samples inside GB_range:", flush=True)


        arr_dicts = []
        for writer_i, npz_file in tqdm(enumerate(self.npz_files)):
            arr_dicts.append(dict(np.load(npz_file)))
        # with open('./arr_dicts_1.0_1.1_1.2.pkl', 'wb') as f:
        #     pickle.dump(arr_dicts, f)
        
        # with open('../arr_dicts_1.0_1.1_1.2.pkl', 'rb') as f:
        #     arr_dicts = pickle.load(f)

        assert len(arr_dicts) == len(self.npz_files) and len(self.npz_files) == 1020
        
        for writer_i in tqdm(range(len(self.npz_files))):
            arr_dict = arr_dicts[writer_i]
            
            for GB_code in list(arr_dict.keys()): # GB_code: GBXXX
                GB = int(GB_code[2:])
                if GB < self.GB_range[0] or GB > self.GB_range[1]:
                    arr_dict.pop(GB_code)
            self.size += len(arr_dict)
            self.arr_list_list.append(list(arr_dict.values()))
        print("There are %d samples inside GB_range totaly in %d npz files." % (self.size, self.total_writer_num), flush=True)

    def get_random_batch_data(self):
        """
        return (num_writer_per_batch * num_entry_per_writer) of samples randomly.
        """
        batch_npy = []
        
        writer_list = list(range(self.total_writer_num))
        random.shuffle(writer_list)
        cnt = 0
        for i in writer_list:
            if len(self.arr_list_list[i]) < self.num_entry_per_writer:
                continue
            batch_npy.extend(copy.deepcopy(random.sample(self.arr_list_list[i], self.num_entry_per_writer)))
            cnt += 1
            if cnt == self.num_writer_per_batch:
                break

        batch_zi = [] 
        batch_len = [] 
        for npy in batch_npy:
            # print(npy_path)
            zi, zi_len = self._processing_zi(npy)
            
            batch_zi.append(zi)
            batch_len.append(zi_len)

        batch_zi_array = format3_zero_pad_to_max_len(batch_zi, self.max_seq_len)  # shape: (batch, max_seq_len, 3)
        batch_len_array = np.array(batch_len, dtype=int)  # shape: (batch, )

        return batch_zi_array, batch_len_array

    def _processing_zi(self, zi):
        """
        process one format3 sample.
        """        
        zi_len = len(zi)
        if zi_len > self.max_seq_len:
            zi_len = self.max_seq_len
            zi = zi[:self.max_seq_len]

        zi[:, 0:2] /= self.scale_factor

        if self.random_scale_factor > 0.0:
            zi = random_scale(zi, self.random_scale_factor)
        if self.augment_stroke_prob > 0.0:
            zi = augment_strokes(zi, self.augment_stroke_prob)
            
        return zi, zi_len



if __name__ == '__main__':
    npz_dir = "/media/tss/S1/hanzi/npz_relative_dist/CASIA_rdp4.0"
    npz_files = [os.path.join(npz_dir, npz) for npz in os.listdir(npz_dir)][:10]

    num_writer_per_batch=8
    num_entry_per_writer=5

    dataset = CHN_Style_DataLoader(
                    npz_files=npz_files,
                    num_writer_per_batch=num_writer_per_batch,
                    num_entry_per_writer=num_entry_per_writer,
                )

    batch_zi_array, batch_len_array = dataset.get_random_batch_data()
    plt.subplots_adjust()
    for i in range(len(batch_len_array)):
        plt.subplot(num_writer_per_batch, num_entry_per_writer, i+1)
        draw_one_hanzi(batch_zi_array[i][:batch_len_array[i]])
    
    plt.show()
