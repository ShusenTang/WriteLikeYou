import os
import time
import random
import ast
import json
import argparse
import numpy as np
import tensorflow as tf

from style_enc_model import Style_Enc_Model
from data import CHN_Style_DataLoader
from model_utils import save_model, reset_graph

NPZ_DIR = "../WriteLikeYouData/npz_relative_dist/CASIA_rdp4.0"

assert os.path.exists(NPZ_DIR)
print("NPZ_DIR:", NPZ_DIR)


def print_save_args(args, save_dir, readme=""):
    os.makedirs(save_dir, exist_ok=True)
    args_dict = vars(args)
    for key in args_dict.keys():
        print(key, "=", args_dict[key])
    with open(os.path.join(save_dir, 'model_config.json'), 'w') as f:
        json.dump(args_dict, f, indent=True)
    if readme != "":
        with open(os.path.join(save_dir, 'readme.txt'),"w") as f:
            f.write(readme)

def evaluate_model(sess, model, dataloader):
    avg_loss = 0.0
    avg_ac_loss = 0.0
    avg_mhe_loss = 0.0
    bn = dataloader.size // dataloader.batch_size 
    for i in range(bn): 
        batch_zi_array, batch_lens = dataloader.get_random_batch_data()
        feed = {
            model.input_data: batch_zi_array,
            model.seq_lens: batch_lens,
        }
        (loss, ac_loss, mhe_loss) = sess.run([model.loss, model.ac_loss, model.mhe_loss], feed)

        avg_loss += loss
        avg_ac_loss += ac_loss
        avg_mhe_loss += mhe_loss
        bn += 1

    avg_loss /= bn
    avg_ac_loss /= bn
    avg_mhe_loss /= bn
    return avg_loss, avg_ac_loss, avg_mhe_loss

def train(sess, model, valid_model, train_dataloader, valid_dataloader, log_root):
    """Train a model."""
    # Setup summary writer.
    os.makedirs(log_root, exist_ok=True)
    train_summary_writer = tf.summary.FileWriter(log_root + "/train_log", sess.graph)
    valid_summary_writer = tf.summary.FileWriter(log_root + "/valid_log")
    saver = tf.train.Saver(tf.global_variables(), max_to_keep=5)

    t_vars = tf.trainable_variables()
    count_t_vars = 0
    for var in t_vars:
        num_param = np.prod(var.get_shape().as_list())
        count_t_vars += num_param
        print('%s %s %d' % (var.name, str(var.get_shape()), num_param))
    print('Total trainable variables %d.' % count_t_vars)

    best_valid_loss = 999999.0
    MIN_LR, DECAY = 0.0000001, 0.9999

    for step in range(1000000):
        start = time.time()
        batch_zi_array, batch_len_array = train_dataloader.get_random_batch_data()

        curr_learning_rate = (model.init_lr - MIN_LR) * (DECAY ** step) + MIN_LR

        feed = {
            model.input_data: batch_zi_array,
            model.seq_lens: batch_len_array,
            model.lr: curr_learning_rate,
        }

        # training
        (_, loss, ac_loss, mhe_loss, summ_str) = sess.run([model.train_op, model.loss, model.ac_loss, model.mhe_loss, model.summ], feed)
        train_summary_writer.add_summary(summ_str, step)
        train_summary_writer.flush()

        # log
        if step % 50 == 0 and step > 0:
            print("Train step %d, lr:%.6f, loss: %.4f, ac_loss: %.4f, mhe_loss: %.4f; train time: %.2fs" % (
                step, curr_learning_rate, loss, ac_loss, mhe_loss, time.time() - start))


        # validation
        if step % 500 == 0 and step > 0:
            start = time.time()
            print("validating...", log_root)
            valid_loss, valid_ac_loss, valid_mhe_loss = evaluate_model(sess, valid_model, valid_dataloader)

            valid_loss_summ = tf.summary.Summary()
            valid_loss_summ.value.add(tag='valid_loss', simple_value=float(valid_loss))
            valid_loss_summ.value.add(tag='valid_ac_loss', simple_value=float(valid_ac_loss))
            valid_loss_summ.value.add(tag='valid_mhe_loss', simple_value=float(valid_mhe_loss))


            print("Best valid loss: %.4f, loss: %.4f, ac_loss: %.4f, mhe_loss: %.4f; valid time: %.2fs" % (
                best_valid_loss, valid_loss, valid_ac_loss, valid_mhe_loss, time.time() - start))

            valid_summary_writer.add_summary(valid_loss_summ, step)
            valid_summary_writer.flush()

            if valid_loss < best_valid_loss:
                best_valid_loss = valid_loss
                print("Better model, best_valid_loss: %.4f" % best_valid_loss)
                save_model(sess, saver, log_root, step)


def trainer(npzs,
            args, 
            train_GB_range,
            valid_GB_range):

    reset_graph()

    model = Style_Enc_Model(                
            init_lr = args.init_lr,
            num_writer_per_batch=args.num_writer_per_batch,
            num_entry_per_writer=args.num_entry_per_writer,
            max_seq_len=args.max_seq_len, 
            rnn_size=args.rnn_size, 
            rnn_layers=args.rnn_layers, 
            embedding_dim = args.embedding_dim,
            ac_softmax_m=args.ac_softmax_m,
            ac_softmax_s=args.ac_softmax_s,
            mhe_lambda=args.mhe_lambda,
            input_keep_prob = args.input_keep_prob,
            output_keep_prob = args.output_keep_prob,
            state_keep_prob = args.state_keep_prob)
    
    valid_model = Style_Enc_Model(                
            init_lr = args.init_lr,
            num_writer_per_batch=args.num_writer_per_batch,
            num_entry_per_writer=args.num_entry_per_writer,
            max_seq_len=args.max_seq_len, 
            rnn_size=args.rnn_size, 
            rnn_layers=args.rnn_layers, 
            embedding_dim = args.embedding_dim,
            ac_softmax_m=args.ac_softmax_m,
            ac_softmax_s=args.ac_softmax_s,
            mhe_lambda=args.mhe_lambda,
            input_keep_prob = 1.0,
            output_keep_prob = 1.0,
            state_keep_prob = 1.0,
            reuse=True,
            is_training=False)

    sess = tf.InteractiveSession()
    sess.run(tf.global_variables_initializer())

    print("train data:")
    train_dataloader = CHN_Style_DataLoader(
            npzs, 
            GB_range = train_GB_range, 
            is_traing=True,

            num_writer_per_batch=args.num_writer_per_batch,
            num_entry_per_writer=args.num_entry_per_writer,
            max_seq_len=args.max_seq_len, 
            scale_factor=200.0,  # for whole data set
            random_scale_factor=args.random_scale_factor,  # only for training
            augment_stroke_prob=args.augment_stroke_prob
            )

    print("valid data:")
    valid_dataloader = CHN_Style_DataLoader(
            npzs, 
            GB_range = valid_GB_range,
            is_traing=False,

            num_writer_per_batch=args.num_writer_per_batch,
            num_entry_per_writer=args.num_entry_per_writer,
            max_seq_len=args.max_seq_len, 
            scale_factor=200.0,  # for whole data set
            random_scale_factor=0.0, 
            augment_stroke_prob=0.0
            )

    train(sess, model, valid_model, train_dataloader, valid_dataloader, args.log_root)


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--init_lr", type=float,  default=0.001, help="init learning rate")
    parser.add_argument("--num_writer_per_batch", type=int, default=64)
    parser.add_argument("--num_entry_per_writer", type=int, default=16)
    parser.add_argument("--max_seq_len", type=int, default=110)
    parser.add_argument("--random_scale_factor", type=float,  default=0.1)
    parser.add_argument("--augment_stroke_prob", type=float,  default=0.1)
    parser.add_argument("--rnn_size", type=int, default=256)
    parser.add_argument("--rnn_layers", type=int, default=1)
    parser.add_argument("--embedding_dim", type=int, default=256)
    parser.add_argument("--ac_softmax_m", type=float, default=0.35)
    parser.add_argument("--ac_softmax_s", type=int, default=30)
    parser.add_argument("--mhe_lambda", type=float, default=0.0)
    parser.add_argument("--input_keep_prob", type=float,  default=1.0)
    parser.add_argument("--output_keep_prob", type=float,  default=1.0)
    parser.add_argument("--state_keep_prob", type=float,  default=1.0)
    parser.add_argument("--log_root", type=str,  default="./models/demo")
    args = parser.parse_args()

    # CUDA_VISIBLE_DEVICES=3 python style_enc_train.py --rnn_layers=4 --ac_softmax_m=0.1 --log_root=./models/0107_layer4_m010

    npz_10 = [os.path.join(NPZ_DIR, "%03d.npz" % n) for n in range(1, 421)]
    npz_11 = [os.path.join(NPZ_DIR, "%d.npz" % n) for n in range(1001, 1301)]
    npz_12 = [os.path.join(NPZ_DIR, "%d.npz" % n) for n in range(501, 801)]
    # npz_C = [os.path.join(NPZ_DIR, "C%03d-f.npz" % n) for n in range(1, 61)]
    npzs = npz_10 + npz_11 + npz_12
    for npz in npzs:
        assert os.path.exists(npz)

    readme = "%d writers: 1.0 %d, 1.1 %d, 1.2 %d." % (len(npzs), len(npz_10), len(npz_11), len(npz_12))
    print(readme)
    readme += "\nGB range: train(201-6763), valid(1-200)"
    
    print_save_args(args, args.log_root, readme=readme)

    trainer(npzs,
            args = args,
            train_GB_range=(201, 6763),
            valid_GB_range=(1, 200))


if __name__ == '__main__':
    main()
