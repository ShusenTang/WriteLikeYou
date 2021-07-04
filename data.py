# -*- coding: utf-8 -*-
import random
import copy
import numpy as np
import matplotlib.pyplot as plt
import os
import sys
from tqdm import tqdm
import pickle

from data_utils import random_scale, augment_strokes, format3_zero_pad_to_max_len, pad_3to5, draw_one_hanzi, to_normal_strokes



class DataLoader(object):
    """Class for loading data."""

    def __init__(self,
                 ref_npz,
                 npz_files,
                 GB_range = (1, 3755), 
                 is_traing=True,
                 batch_size=256,
                 max_seq_len=100, 
                 scale_factor=200.0,  # for whole data set
                 random_scale_factor=0.0,  # only for training
                 augment_stroke_prob=0.0,  # only for training
                 style_input_num = 10,
                 ):
                
        self.ref_npz = ref_npz
        self.npz_files = npz_files
        self.total_writer_num = len(npz_files)
        print("%d writers in total:" % self.total_writer_num)
        self.writer_i2s = [d.split("/")[-1] for d in npz_files]
        self.writer_s2i = {self.writer_i2s[i]:i for i in range(self.total_writer_num)}
        

        print(" ".join([d.split("/")[-1] for d in npz_files]))
        print("reference font:", ref_npz)

        self.GB_range = GB_range
        print("GB range:", GB_range)
        self.is_traing = is_traing
        self.batch_size = batch_size  # minibatch size
        self.max_seq_len = max_seq_len
        self.scale_factor = scale_factor  # divide offsets by this factor

        if not is_traing:
            assert random_scale_factor == 0
            assert augment_stroke_prob == 0
        self.random_scale_factor = random_scale_factor  # data augmentation method
        self.augment_stroke_prob = augment_stroke_prob  # data augmentation method
        self.style_input_num = style_input_num

        self._get_npz_data()
        self.idxes = list(range(self.size))


    def _get_npz_data(self):
        """
        load all npz data, including data outside the GB_range.
        ref_arr_dict: dict of reference input. key: GB_code(e.g., "GB100"), value: numpy array
        arr_dict_list: list of dict, arr_dict_list[i] contains all npy dict in npz_files[i]. 
        npy_idxes: indexes of samples which inside the GB_range: (writer_i, GB_code)
        GB_list_list: legal GBcode for each writer, GB_list_list[i] contains the legal GB_codes of the i'th writer
        """
        self.ref_arr_dict = dict(np.load(self.ref_npz))
        self.arr_dict_list = []
        self.npy_idxes = []
        self.GB_list_list = []
        print("reading data:", flush=True)

        if len(self.npz_files) == 1020:
            print("load all data from arr_dicts_1.0_1.1_1.2.pkl directly:", flush=True)
            with open('./arr_dicts_1.0_1.1_1.2.pkl', 'rb') as f:
                arr_dicts = pickle.load(f)

        elif len(self.npz_files) == 60 and self.is_traing == False:
            print("load all data from arr_dicts_competition.pkl directly:", flush=True)
            with open('./arr_dicts_competition.pkl', 'rb') as f:
                arr_dicts = pickle.load(f)
        else:
            arr_dicts = []
            for writer_i, npz_file in tqdm(enumerate(self.npz_files)):
                arr_dicts.append(dict(np.load(npz_file)))
            # with open('./arr_dicts_1.0_1.1_1.2.pkl', 'wb') as f:
            #     pickle.dump(arr_dicts, f)
        
        assert len(arr_dicts) == len(self.npz_files)
        
        print("processing data:")
        for writer_i in tqdm(range(len(self.npz_files))):
            arr_dict = arr_dicts[writer_i]
            self.arr_dict_list.append(arr_dict)

            GB_list = []  # inside range
            valid_GB = list(arr_dict.keys())
            for GB_code in range(self.GB_range[0], self.GB_range[1] + 1):
                GB_code_str = "GB" + str(GB_code)  # GB_code_str: GBXXX
                if GB_code_str in valid_GB:
                    self.npy_idxes.append((writer_i, GB_code))
                    GB_list.append(GB_code)

            self.GB_list_list.append(GB_list.copy())

        self.size = len(self.npy_idxes)
        print("There are %d samples in total." % (self.size), flush=True)


    def shuffle_idxes(self):
        # print("Shuffle the idexes!")
        random.shuffle(self.idxes)


    def yield_batch_data(self, batch_size=None):
        """batch data with padding: targets(format5), content inputs(format3), style inputs(format3)
        """
        if self.is_traing:
            self.shuffle_idxes()
        if batch_size is None: batch_size = self.batch_size        
        
        bn = self.size // batch_size
        for b in range(bn):
            target_zi_list = []
            target_zi_len_list = []
            content_zi_list = []
            content_zi_len_list = []
            style_zi_list = []
            style_zi_len_list = []
            for idx in self.idxes[batch_size*b: batch_size*b + batch_size]:
                npy_idx = self.npy_idxes[idx]

                target_zi, target_zi_len, content_zi, content_zi_len, style_zies, style_zi_lens = self._get_input_sample(npy_idx)
                
                target_zi_list.append(target_zi)
                target_zi_len_list.append(target_zi_len)
                content_zi_list.append(content_zi)
                content_zi_len_list.append(content_zi_len)
                style_zi_list.extend(style_zies)
                style_zi_len_list.extend(style_zi_lens)


            target_zi_array = pad_3to5(target_zi_list, self.max_seq_len)  # shape: (batch, max_seq_len + 1, 5)
            target_zi_len_array = np.array(target_zi_len_list, dtype=int) # shape: (batch, )
            content_zi_array = format3_zero_pad_to_max_len(content_zi_list, self.max_seq_len)  # shape: (batch, max_seq_len, 3)
            content_zi_len_array = np.array(content_zi_len_list, dtype=int) # shape: (batch, )
            style_zi_array = format3_zero_pad_to_max_len(style_zi_list, self.max_seq_len)  # shape: (batch * style_input_num, max_seq_len, 3)
            style_zi_len_array = np.array(style_zi_len_list, dtype=int) # shape: (batch * style_input_num, )

            yield target_zi_array, target_zi_len_array, content_zi_array, content_zi_len_array, style_zi_array, style_zi_len_array


    def _get_input_sample(self, target_npy_idx):
        """
        load an sample by the target index, contains three parts: target(format3), content input, multiple style inputs.
        target_npy_idx: (writer_i, GB_code)
        """
        writer_i = target_npy_idx[0]
        GB_code_key = "GB" + str(target_npy_idx[1])
        content_zi, content_zi_len = self._processing_zi(copy.deepcopy(self.ref_arr_dict[GB_code_key])) # deepcopy!!!
        target_zi, target_zi_len = self._processing_zi(copy.deepcopy(self.arr_dict_list[writer_i][GB_code_key])) # deepcopy!!!
 
        style_zies = []
        style_zi_lens = []
        for GB in random.sample(self.GB_list_list[writer_i], self.style_input_num):
            style_zi, style_zi_len = self._processing_zi(copy.deepcopy(self.arr_dict_list[writer_i]["GB"+str(GB)]))
            style_zies.append(style_zi)
            style_zi_lens.append(style_zi_len)

        return target_zi, target_zi_len, content_zi, content_zi_len, style_zies, style_zi_lens


    def _processing_zi(self, zi):
        """
        processing one format3 sample: scale, pad or clip, augment
        """
        zi_len = len(zi)

        if zi_len > self.max_seq_len:
            zi_len = self.max_seq_len
            zi = zi[:self.max_seq_len]
        elif zi_len == 0:
            assert False

        zi[:, 0:2] /= self.scale_factor

        if self.random_scale_factor > 0.0:
            zi = random_scale(zi, self.random_scale_factor)
        if self.augment_stroke_prob > 0.0:
            zi = augment_strokes(zi, self.augment_stroke_prob)
        
        return zi, zi_len


    def get_specified_data(self, writer_i=None, GB_code=None, verbose=False):
        """
        return data specified by index, for testing
        """
        if writer_i == None:
            writer_i = random.randint(0, self.total_writer_num - 1)
        if GB_code == None:
            GB_code = random.choice(self.GB_list_list[writer_i])
        
        npy_idx = (writer_i, GB_code) 
        if verbose:
            print("target npy idx: ", npy_idx)

        target_zi, target_zi_len, content_zi, content_zi_len, style_zies, style_zi_lens = self._get_input_sample(npy_idx)
        
        target_zi_array = pad_3to5([target_zi], self.max_seq_len)  # shape: (1, max_seq_len + 1, 5)
        target_zi_len_array = np.array([target_zi_len], dtype=int) # shape: (1, )
        content_zi_array = format3_zero_pad_to_max_len([content_zi], self.max_seq_len)  # shape: (1, max_seq_len, 3)
        content_zi_len_array = np.array([content_zi_len], dtype=int) # shape: (1, )
        style_zi_array = format3_zero_pad_to_max_len(style_zies, self.max_seq_len)  # shape: (1 * style_input_num, max_seq_len, 3)
        style_zi_len_array = np.array(style_zi_lens, dtype=int) # shape: (1 * style_input_num, )

        return target_zi_array, target_zi_len_array, content_zi_array, content_zi_len_array, style_zi_array, style_zi_len_array



if __name__ == '__main__':
    ref_npz = "/media/tss/S1/hanzi/npz_relative_dist/rdp4.0/mean.npz"
    npz_dir = "/media/tss/S1/hanzi/npz_relative_dist/CASIA_rdp4.0"
    npz_files = [os.path.join(npz_dir, npz) for npz in os.listdir(npz_dir)][:10]


    dataset = DataLoader(
                 ref_npz,
                 npz_files,
                 GB_range = (1, 40), 
                 is_traing=True,
                 batch_size=16,
                 style_input_num = 10)


    for target_zi_array, target_zi_len_array, content_zi_array, content_zi_len_array, style_zi_array, style_zi_len_array in dataset.yield_batch_data():
        plt.figure(figsize=(50, 4))
        plt.subplots_adjust()
        plt.subplot(2, 10, 1)
        draw_one_hanzi(content_zi_array[0][:content_zi_len_array[0]])
        
        plt.subplot(2, 10, 2)
        draw_one_hanzi(to_normal_strokes(target_zi_array[0]))

        for i in range(dataset.style_input_num):
            plt.subplot(2, 10, 11+i)
            draw_one_hanzi(style_zi_array[i][:style_zi_len_array[i]])
        
        plt.show()
        break

    npy_idx = dataset.npy_idxes[dataset.idxes[0]]
    print(npy_idx)


