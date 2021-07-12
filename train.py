# -*- coding: utf-8 -*-
import json
import os
import time
import six
import re
import ast

import numpy as np
import tensorflow as tf

from model_utils import save_model, reset_graph
from model import Model, get_default_hparams, copy_hparams
import data

debug = True

NPZ_DIR = "../WriteLikeYouData/npz_relative_dist/CASIA_rdp4.0"
REF_NPZ = "../WriteLikeYouData/npz_relative_dist/mean.npz"

assert os.path.exists(NPZ_DIR) and os.path.exists(REF_NPZ)
print("NPZ_DIR:", NPZ_DIR)
print("REF_NPZ:", REF_NPZ)

tf.logging.set_verbosity(tf.logging.INFO)


FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_string(
    'hparams', '',
    'Pass in comma-separated key=value pairs such as '
    '\'save_every=40,decay_rate=0.99\' '
    '(no whitespace) to be read into the HParams object defined in model.py')


def load_hps(model_dir):
    """Loads environment for inference mode"""
    model_params = get_default_hparams()
    with tf.gfile.Open(os.path.join(model_dir, 'model_config.json'), 'r') as f:
        model_params.parse_json(f.read())

    for key, val in six.iteritems(model_params.values()):
        print('%s = %s' % (key, str(val)))
    return model_params


def load_checkpoint(sess, checkpoint_path):
    saver = tf.train.Saver(tf.global_variables())
    ckpt = tf.train.get_checkpoint_state(checkpoint_path)
    tf.logging.info('Loading model %s.', ckpt.model_checkpoint_path)
    saver.restore(sess, ckpt.model_checkpoint_path)


def evaluate_model(sess, model, dataloader, loss_str_list, curr_cycle_loss_weight=1.0):
    loss_num = len(loss_str_list)
    avg_losses = [0.0 for _ in range(loss_num)]
    bn = 0
    for target_zi, target_zi_len, content_zi, content_zi_len, style_zi, style_zi_len in dataloader.yield_batch_data():
        feed = {
                model.enc1_input_data: content_zi,
                model.enc1_seq_lens: content_zi_len,
                model.enc2_input_data: style_zi,
                model.enc2_seq_lens: style_zi_len,
                model.dec_input_data: target_zi,
                model.dec_seq_lens: target_zi_len,
                model.curr_cycle_loss_weight: curr_cycle_loss_weight
            }
        losses = sess.run([getattr(model, loss) for loss in loss_str_list], feed)

        for i in range(loss_num):
            avg_losses[i] += losses[i]
        bn += 1

    for i in range(loss_num):
        avg_losses[i] /= bn
    return avg_losses


def train(sess, model, valid_model, train_dataloader, valid_dataloader, print_every=None):
    """Train a model."""
    # Setup summary writer.
    train_summary_writer = tf.summary.FileWriter(model.hps.log_root + "/train_log", sess.graph)
    valid_summary_writer = tf.summary.FileWriter(model.hps.log_root + "/valid_log")
    saver = tf.train.Saver(tf.global_variables(), max_to_keep=5)

    t_vars = tf.trainable_variables()
    count_t_vars = 0
    for var in t_vars:
        num_param = np.prod(var.get_shape().as_list())
        count_t_vars += num_param
        tf.logging.info('%s %s %i', var.name, str(var.get_shape()), num_param)
    tf.logging.info('Total trainable variables %i.', count_t_vars)

    # setup eval stats
    best_valid_cost = 100000000.0  # set a large init value

    # main train loop
    hps = model.hps  # hyparameter
    if print_every is None:
        print_every = int(hps.save_every / 10)
    step = 0
    for epoch in range(100000):
        for target_zi, target_zi_len, content_zi, content_zi_len, style_zi, style_zi_len in train_dataloader.yield_batch_data():
            step += 1
            start = time.time()

            curr_learning_rate = ((hps.learning_rate - hps.min_learning_rate) *
                                (hps.decay_rate) ** step + hps.min_learning_rate)
            curr_cycle_loss_weight = 1.0 - (hps.cycle_loss_decay ** step)

            feed = {
                model.enc1_input_data: content_zi,
                model.enc1_seq_lens: content_zi_len,
                model.enc2_input_data: style_zi,
                model.enc2_seq_lens: style_zi_len,
                model.dec_input_data: target_zi,
                model.dec_seq_lens: target_zi_len,
                model.lr: curr_learning_rate,
                model.curr_cycle_loss_weight: curr_cycle_loss_weight
            }

            # training
            (_, cost, r_cost, Ls, Lp, content_cycle_loss, style_cycle_loss, summ_str) = sess.run([model.train_op, 
                model.cost, model.r_cost, model.Ls, model.Lp, model.content_cycle_loss, model.style_cycle_loss, model.summ], feed)
            train_summary_writer.add_summary(summ_str, step)
            train_summary_writer.flush()

            if step % print_every == 0 and step > 0:
                message = "Train step %d (epoch:%d), lr:%.6f, cost: %.4f, r_cost:%.4f, Ls: %.4f, Lp:%.4f, c_c:%.4f, s_c:%.4f(w: %.4f); time: %.2fs"
                print(message % (
                    step, epoch, curr_learning_rate, cost, r_cost, Ls, Lp, content_cycle_loss, style_cycle_loss, curr_cycle_loss_weight, time.time() - start))


            # validation
            if step % hps.save_every == 0 and step > 0:
                start = time.time()
                print("validating...", model.hps.log_root)
                loss_str_list = ["cost", "r_cost", "Ls", "Lp", "content_cycle_loss", "style_cycle_loss"]
                valid_avg_losses = evaluate_model(sess, valid_model, valid_dataloader, loss_str_list)


                message = "best: %.4f"
                for loss_i, loss_str in enumerate(loss_str_list):
                    valid_summ = tf.summary.Summary()
                    valid_summ.value.add(tag='valid_' + loss_str, simple_value=float(valid_avg_losses[loss_i]))
                    valid_summary_writer.add_summary(valid_summ, step)
                    message += ", l_" + str(loss_i) + ": %.4f"

                message += "; valid time: %.2fs"
                output_values = [best_valid_cost] + valid_avg_losses + [time.time() - start]

                print(message % tuple(output_values))
                valid_summary_writer.flush()

                if valid_avg_losses[0] < best_valid_cost:
                    best_valid_cost = valid_avg_losses[0]
                    print("Better model, best_valid_cost: %.4f" % best_valid_cost)
                    save_model(sess, saver, model.hps.log_root, step)


def trainer(model_params):
    np.set_printoptions(precision=8, edgeitems=6, linewidth=200, suppress=True) 

    tf.logging.info('Training a model:')
    tf.logging.info('Hyperparams:')
    for key, val in six.iteritems(model_params.values()):
        tf.logging.info('%s = %s', key, str(val))

    npz_10 = [os.path.join(NPZ_DIR, "%03d.npz" % n) for n in range(1, 421)]
    npz_11 = [os.path.join(NPZ_DIR, "%d.npz" % n) for n in range(1001, 1301)]
    npz_12 = [os.path.join(NPZ_DIR, "%d.npz" % n) for n in range(501, 801)]
    # npz_C = [os.path.join(NPZ_DIR, "C%03d-f.npz" % n) for n in range(1, 61)]
    npzs = npz_10 + npz_11 + npz_12
    for npz in npzs:
        assert os.path.exists(npz)

    eval_model_params = copy_hparams(model_params)
    eval_model_params.is_training = False
    eval_model_params.input_keep_prob = 1.0
    eval_model_params.output_keep_prob = 1.0
    eval_model_params.state_keep_prob = 1.0

    reset_graph()
    model = Model(model_params)
    eval_model = Model(eval_model_params, reuse=True)

    sess = tf.InteractiveSession()
    sess.run(tf.global_variables_initializer())

    if model_params.pretrain_content_enc != "":
        print("\nLoad pretrained content encoder from %s " % model_params.pretrain_content_enc)
        variables_to_resotre = tf.contrib.framework.get_variables_to_restore(include=["Content_Encoder"])
        for var in variables_to_resotre:
            print(var.name)
        saver = tf.train.Saver(variables_to_resotre)
        ckpt = tf.train.get_checkpoint_state(model_params.pretrain_content_enc)
        saver.restore(sess, ckpt.model_checkpoint_path)

    if model_params.pretrain_style_enc != "":
        print("\nLoad pretrained style encoder from %s " % model_params.pretrain_style_enc)
        variables_to_resotre = tf.contrib.framework.get_variables_to_restore(include=["Style_Encoder"])
        for var in variables_to_resotre:
            print(var.name)
        saver = tf.train.Saver(variables_to_resotre)
        ckpt = tf.train.get_checkpoint_state(model_params.pretrain_style_enc)
        saver.restore(sess, ckpt.model_checkpoint_path)
        print("\n\n")

    # Write config file to json file.
    tf.gfile.MakeDirs(model_params.log_root) 
    with tf.gfile.Open(
            os.path.join(model_params.log_root, 'model_config.json'), 'w') as f:
        json.dump(model_params.values(), f, indent=True)


    print("\nTrain dataloader:")
    train_dataloader = data.DataLoader(
                 ref_npz=REF_NPZ,
                 npz_files=npzs,
                 GB_range = (201, 6763), # start from 1
                 is_traing=True,
                 batch_size=model_params.batch_size,
                 max_seq_len=model_params.max_seq_len, 
                 scale_factor=200.0,  # for whole data set
                 random_scale_factor=model_params.random_scale_factor,  # only for training
                 augment_stroke_prob=model_params.augment_stroke_prob,  # only for training
                 style_input_num = model_params.style_input_num)    
    
    print("\nValidation dataloader:")
    valid_dataloader = data.DataLoader(
                 ref_npz=REF_NPZ,
                 npz_files=npzs,
                 GB_range = (1, 200), 
                 is_traing=False,
                 batch_size=model_params.batch_size,
                 max_seq_len=model_params.max_seq_len, 
                 scale_factor=200.0,  # for whole data set
                 random_scale_factor=0.0,  # only for training
                 augment_stroke_prob=0.0,  # only for training
                 style_input_num = model_params.style_input_num)

    # print("\n\n\n", len(train_set.strokes))
    train(sess, model, eval_model, train_dataloader, valid_dataloader)


def main(unused_argv):
    """Load model params, save config file and start trainer."""
    model_params = get_default_hparams() 
    if FLAGS.hparams:
        model_params.parse(FLAGS.hparams) 

    trainer(model_params)  


def console_entry_point():
    tf.app.run(main)

if __name__ == '__main__':
    console_entry_point()
