# -*- coding: utf-8 -*-
import numpy as np
import os
import ast
import argparse
from tqdm import tqdm
import tensorflow as tf

import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt

import train 
import model
import gmm
import data
from data_utils import to_normal_strokes, draw_one_hanzi

NPZ_DIR = "/data1/tangss/npz_relative_dist/CASIA_rdp4.0"
REF_NPZ = "/data1/tangss/npz_relative_dist/rdp4.0/mean.npz"
assert os.path.exists(NPZ_DIR) and os.path.exists(REF_NPZ)
print("NPZ_DIR:", NPZ_DIR)
print("REF_NPZ:", REF_NPZ)


# models_root = './ft_models'
# save_root = './result'
# writer = 19
# model_name = "ft_C%03d_100" % writer # "1128_enc2size128_layer2_stylecycle10_top10" # "1115_mixture10_styleinput20"
# writers = ["C%03d-f"%writer] # for i in range(1, 61)][:1]
# npzs = [os.path.join(NPZ_DIR, "%s.npz" % i) for i in writers]
# if "ft" in model_name:
#     GB_range = (1, 400)
#     generate_range = (1, 100)
#     print("finetune模型, 生成样本区间为", generate_range)

models_root = './ft_models'
save_root = './result'
model_name = "ft_C001_100" # "1128_enc2size128_layer2_stylecycle10_top10" # "1115_mixture10_styleinput20"
writers = ["C%03d-f"%i for i in range(1, 61)][:1]
npzs = [os.path.join(NPZ_DIR, "%s.npz" % i) for i in writers]
GB_range = (201, 600)
generate_range = (201, 300)
GB_range = (1, 600)
generate_range = (1, 300)


print("model name: ", model_name)
model_dir = os.path.join(models_root, model_name)
pic_save_dir = os.path.join(save_root, model_name, "pic")
fake_pic_save_dir = os.path.join(save_root, model_name, "fake_pic")
npz_save_dir = os.path.join(save_root, model_name, "npz")
os.makedirs(pic_save_dir, exist_ok=True)
os.makedirs(npz_save_dir, exist_ok=True)


def testing_model(sess, testmodel, content_zi, content_zi_len, style_zi, style_zi_len):
    feed = {
        testmodel.enc1_input_data: content_zi,
        testmodel.enc1_seq_lens: content_zi_len,
        testmodel.enc2_input_data: style_zi,
        testmodel.enc2_seq_lens: style_zi_len,
    }

    output = sess.run([testmodel.pi, testmodel.mu1, testmodel.mu2, testmodel.sigma1,
                       testmodel.sigma2, testmodel.corr, testmodel.pen,
                       testmodel.timemajor_alignment_history,
                       testmodel.style_timemajor_alignment_history
                       ],
                      feed)
    gmm_params = output[:7]
    time_attn_align_hist = output[7:]
    return gmm_params, time_attn_align_hist


def sample_from_params(params, temp=0.1, greedy=False, sample_mode="max"):
    # 前六个shape=(batch_size * max_seq_le, num_mixture), 最后两个为(batch_size * max_seq_le, 3)
    [o_pi, o_mu1, o_mu2, o_sigma1, o_sigma2, o_corr, o_pen] = params

    max_len = o_pi.shape[0]
    num_mixture = o_pi.shape[1]

    strokes = np.zeros((max_len, 5), dtype=np.float32)

    for step in range(max_len):
        eos = [0, 0, 0]
        eos[np.argmax(o_pen[step])] = 1
        if sample_mode == "mean":
            next_x1 = 0
            next_x2 = 0
            for mixture in range(num_mixture):
                x1, x2 = gmm.sample_gaussian_2d(o_mu1[step][mixture], o_mu2[step][mixture],
                                                o_sigma1[step][mixture], o_sigma2[step][mixture],
                                                o_corr[step][mixture], np.sqrt(temp), greedy)
                next_x1 += x1 * o_pi[step][mixture]
                next_x2 += x2 * o_pi[step][mixture]
        elif sample_mode == "max":
            mixture = np.argmax(o_pi[step])
            next_x1, next_x2 = gmm.sample_gaussian_2d(o_mu1[step][mixture], o_mu2[step][mixture],
                                                o_sigma1[step][mixture], o_sigma2[step][mixture],
                                                o_corr[step][mixture], np.sqrt(temp), greedy)
            
        else:
            assert False
        # next_x = [next_x1, next_x2]
        strokes[step, :] = [next_x1, next_x2, eos[0], eos[1], eos[2]]
    strokes = to_normal_strokes(strokes)
    # print(strokes[0])
    return strokes


def draw_ref_real_fake(delta_ref_stroke, delta_gt_stroke, delta_stroke, lim=None, style_name='', axis='on',
                       save_path=None):
    plt.figure(figsize=(6, 2))
    plt.subplot(131)
    draw_one_hanzi(delta_ref_stroke, lim, axis=axis, title='reference')
    plt.subplot(132)
    draw_one_hanzi(delta_gt_stroke, lim, axis=axis, title=style_name + ' real')
    plt.subplot(133)
    draw_one_hanzi(delta_stroke, lim, axis=axis, title=style_name + ' fake')
    if save_path is not None:
        plt.savefig(save_path)

def save_fake(delta_stroke, save_path, lim=None, axis='off', linewidth=3):
    stroke = delta_stroke.copy()
    points_num = delta_stroke.shape[0]
    # 转换成绝对坐标
    low_tri_matrix = np.tril(np.ones((points_num, points_num)), 0)
    stroke[:, :2] = np.matmul(low_tri_matrix, delta_stroke[:, :2])

    plt.figure(figsize=(3, 3))
    # plt.title(title + "\npts:%d, strokes:%d" % (points_num, np.sum(delta_stroke[:, 2])), fontdict={"fontsize": fontsize})
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
    plt.savefig(save_path, dpi=100)
    plt.close()

def generate_result(sess, testmodel, test_dataloader, writer_i = None, GB_code = None, sample_mode="max"):
    tmp = 10
    while tmp:
        target_zi, target_zi_len, content_zi, content_zi_len, style_zi, style_zi_len \
            = test_dataloader.get_specified_data(writer_i = writer_i, GB_code = GB_code, verbose=False)

        gmm_params, align_hist = testing_model(sess, testmodel, content_zi, content_zi_len, style_zi, style_zi_len)
        fake = sample_from_params(gmm_params, temp=1.0, greedy=True, sample_mode=sample_mode)
        tmp -= 1
        if np.sum(fake[:, 2]) > 0:
            break
        
    real = to_normal_strokes(target_zi[0][1:])
    ref = content_zi[0][:content_zi_len[0]]
    # style_zi_no_pad = [style_zi[i, :style_zi_len[i], :] for i in range(int(hps.style_input_num))]

    return ref, real, fake # format3

def main():
    hps = train.load_hps(model_dir)
    # construct model here:
    train.reset_graph()
    trained_model = model.Model(hps)

    test_hps = model.copy_hparams(hps)
    test_hps.input_keep_prob = 1.0
    test_hps.output_keep_prob = 1.0
    test_hps.state_keep_prob = 1.0
    test_hps.batch_size=1
    test_hps.is_training = False
    test_hps.is_testing = True
    test_model = model.Model(test_hps, reuse=True)

    sess = tf.InteractiveSession()
    sess.run(tf.global_variables_initializer())
    train.load_checkpoint(sess, model_dir)

    print("\nTest dataloader:")
    test_dataloader = data.DataLoader(
                ref_npz=REF_NPZ,
                npz_files=npzs,
                GB_range = GB_range, # 左闭右闭, 且从1开始
                is_traing=False,
                batch_size=test_hps.batch_size,
                max_seq_len=test_hps.max_seq_len, 
                scale_factor=200.0,  # for whole data set
                random_scale_factor=0.0,  # only for training
                augment_stroke_prob=0.0,  # only for training
                style_input_num = test_hps.style_input_num)  

    print("testing %s, generate range (%d, %d)" % (model_name, generate_range[0], generate_range[1]))
    for writer_i in tqdm(range(test_dataloader.total_writer_num)):
        os.makedirs(os.path.join(pic_save_dir, writers[writer_i]), exist_ok=True)
        os.makedirs(os.path.join(fake_pic_save_dir, writers[writer_i]), exist_ok=True)
        npy_dict = {}
        for GB_code in range(generate_range[0], generate_range[1]+1):
            try:
                ref, real, fake = generate_result(sess, test_model, test_dataloader, 
                                                    writer_i, GB_code, sample_mode="max")
            except KeyError:
                continue

            pic_save_path = os.path.join(pic_save_dir, writers[writer_i], str(GB_code)+".png")
            draw_ref_real_fake(ref, real, fake, lim=(-1, 1), style_name=writers[writer_i], save_path=pic_save_path)
            fake_pic_save_path = os.path.join(fake_pic_save_dir, writers[writer_i], str(GB_code)+".png")
            save_fake(fake, fake_pic_save_path, lim=(-1, 1), axis='off')
            
            fake[:, :2] *= 200.0
            npy_dict["GB"+str(GB_code)] = fake.copy()
            plt.close()

        np.savez(os.path.join(npz_save_dir, writers[writer_i]+".npz"), **npy_dict)



if __name__ == '__main__':
    main()
