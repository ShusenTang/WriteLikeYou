import os
import random
import numpy as np
from tqdm import tqdm 
import matplotlib.pyplot as plt


def npys2npz(npys_dir, save_root):
    """Read all npy files of a font and save them in an npz file, use GBxxx as the key
    npys_dir: folder path of a font
    """
    print("Start processing %s" % npys_dir, flush=True)
    arrs = {}
    count = 0
    for npy in tqdm(os.listdir(npys_dir)):
        GB_str = "GB" + npy.split("GB")[-1][:-4]
        npy_path = os.path.join(npys_dir, npy)
        arr = np.load(npy_path)
        arrs[GB_str] = arr
        if(len(arr) == 0):
            count += 1
    save_path = os.path.join(save_root, npys_dir.split("/")[-1] + ".npz")
    np.savez(save_path, **arrs)
    print("Save at %s done. %d samples with length equal to 0" % (save_path, count))


def random_scale(strokes, random_scale_factor):
    """Augment data by stretching x and y axis randomly [1-e, 1+e]."""
    x_scale_factor = (np.random.random() - 0.5) * 2 * random_scale_factor + 1.0
    y_scale_factor = (np.random.random() - 0.5) * 2 * random_scale_factor + 1.0
    result = np.copy(strokes)
    result[:, 0] *= x_scale_factor
    result[:, 1] *= y_scale_factor
    return result


def augment_strokes(strokes, prob=0.0):
    """Perform data augmentation by randomly dropping out strokes."""
    # drop each point within a line segments with a probability of prob
    # note that the logic in the loop prevents points at the ends to be dropped.
    result = []
    prev_stroke = [0, 0, 1]
    count = 0
    stroke = [0, 0, 1]  # Added to be safe.
    for i in range(len(strokes)):
        candidate = [strokes[i][0], strokes[i][1], strokes[i][2]]
        if candidate[2] == 1 or prev_stroke[2] == 1:  # current or previous point is the end of stroke
            count = 0
        else:
            count += 1
        urnd = np.random.rand()  # uniform random variable
        if candidate[2] == 0 and prev_stroke[2] == 0 and count > 2 and urnd < prob:  # drop
            stroke[0] += candidate[0]
            stroke[1] += candidate[1]
        else:
            stroke = candidate
            prev_stroke = stroke
            result.append(stroke)
    return np.array(result)


def format3_zero_pad_to_max_len(hanzi_list, max_len):
    """
    input:
        - hanzi_list: list of hanzi, shape of each hanzi is (seq_len, 3)
    return:
        - result: numpy array, shape (len(hanzi_list), max_len, 3)
    """
    num = len(hanzi_list)
    result = np.zeros((num, max_len, 3), dtype=float)
    for i in range(num):
        l = min(len(hanzi_list[i]), max_len)
        result[i, :l, :] = hanzi_list[i][:l, :]
    return result


def to_normal_strokes(big_stroke):
    """Convert from stroke-5 format (without S0) back to stroke-3."""
    l = 0
    for i in range(len(big_stroke)):
        if big_stroke[i, 4] > 0:
            l = i
            break
    if l == 0:
        l = len(big_stroke)
    result = np.zeros((l, 3))
    result[:, 0:2] = big_stroke[0:l, 0:2]
    result[:, 2] = big_stroke[0:l, 3]
    return result


def pad_3to5(hanzi_list, max_len):
    """Pad the list of hanzi to be format-5. Padded hanzi shape:(max_len + 1, 5).
    input:
        - hanzi_list: list of hanzi, shape of each hanzi is (seq_len, 3)

    return:
        - result: numpy array, shape (len(hanzi_list), max_len + 1, 5)
    """
    num = len(hanzi_list)
    result = np.zeros((num, max_len + 1, 5), dtype=float)
    for i in range(num):
        l = len(hanzi_list[i])
        assert l <= max_len
        result[i, 0:l, 0:2] = hanzi_list[i][:, 0:2]
        result[i, 0:l, 3] = hanzi_list[i][:, 2]
        result[i, 0:l, 2] = 1 - result[i, 0:l, 3]
        result[i, l:, 4] = 1
        result[i, 1:, :] = result[i, :-1, :]

        # setting S_0 (0, 0, 1, 0, 0) from paper
        result[i, 0, :] = 0
        result[i, 0, 2] = 1
        # result[i, 0, 3] = 0
        # result[i, 0, 4] = 0
    return result


def draw_one_hanzi(delta_stroke, lim=(-1, 1), axis='on', title='', linewidth=3, fontsize=10):
    """Input is an numpy array of relative coordinates, shape: (point_num, 3)"""
    stroke = delta_stroke.copy()
    points_num = delta_stroke.shape[0]
    # Convert to absolute coordinates
    low_tri_matrix = np.tril(np.ones((points_num, points_num)), 0)
    stroke[:, :2] = np.matmul(low_tri_matrix, delta_stroke[:, :2])

    # plt.figure(figsize=(3, 3))
    plt.title(title + "\npts:%d, strokes:%d" % (points_num, np.sum(delta_stroke[:, 2])), fontdict={"fontsize": fontsize})
    if lim is not None:
        plt.xlim(lim[0], lim[1])
        plt.ylim(lim[0], lim[1])
    pre_i = 0
    for i in range(stroke.shape[0]):
        if stroke[i][2] == 1:
            plt.plot(stroke[pre_i:i + 1, 0], stroke[pre_i:i + 1, 1], color='black', linewidth=linewidth)
            pre_i = i + 1
    plt.axis(axis)
    plt.gca().invert_yaxis()

    

