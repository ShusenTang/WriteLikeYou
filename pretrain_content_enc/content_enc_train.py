import os
import time
import random
import numpy as np
import tensorflow as tf

from content_enc_model import Content_Enc_Model
from data import CHN_Reco_DataLoader
from model_utils import save_model, reset_graph

OLHWDB_root = "../../CASIA_rdp4.0"

def get_data_dirs(OLHWDB10_dir, OLHWDB11_dir):
    """
    train: writer 001~420 + 1001~1240 (part of OLHWDB1.0 + OLHWDB1.1). 660 writers in total
    valid: 1241~1270, 30 writers
    test: 1271~1300, 30 writers
    """
    DB10 = [os.path.join(OLHWDB10_dir, d) for d in os.listdir(OLHWDB10_dir)]
    DB11 = [os.path.join(OLHWDB11_dir, d) for d in os.listdir(OLHWDB11_dir)]
    DB10.sort()
    DB11.sort()
    train_dirs = DB10 + DB11[:240]
    valid_dirs = DB11[240:270]
    test_dirs = DB11[270:]
    return train_dirs, valid_dirs, test_dirs 

def evaluate_model(sess, model, dataloader):
    avg_loss = 0.0
    avg_acc = 0.0
    bn = 0
    for batch_zi_array, batch_lens, batch_labels in dataloader.yield_batch_data():
        feed = {
            model.input_data: batch_zi_array,
            model.seq_lens: batch_lens,
            model.labels: batch_labels
        }
        (loss, acc) = sess.run([model.loss, model.acc], feed)

        avg_loss += loss
        avg_acc += acc
        bn += 1

    avg_loss /= bn
    avg_acc /= bn
    return (avg_loss, avg_acc)

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

    best_valid_acc = 0.0
    MIN_LR, DECAY = 0.0000001, 0.9999

    step = 0
    for epoch in range(100000):
        for batch_zi_array, batch_len_array, batch_label_array in train_dataloader.yield_batch_data():
            step += 1
            start = time.time()
            curr_learning_rate = (model.init_lr - MIN_LR) * (DECAY ** step) + MIN_LR

            feed = {
                model.input_data: batch_zi_array,
                model.seq_lens: batch_len_array,
                model.labels: batch_label_array,
                model.lr: curr_learning_rate,
            }

            # training
            (_, loss, acc, summ_str) = sess.run([model.train_op, model.loss, model.acc, model.summ], feed)
            train_summary_writer.add_summary(summ_str, step)
            train_summary_writer.flush()

            # log
            if step % 20 == 0 and step > 0:
                print("Train step %d (epoch:%d), lr:%.6f, loss: %.4f, acc:%.4f; train time: %.2fs" % (
                    step, epoch, curr_learning_rate, loss, acc, time.time() - start))

            # validation
            if step % 200 == 0 and step > 0:
                start = time.time()
                print("validating...", log_root)
                (valid_loss, valid_acc) = evaluate_model(sess, valid_model, valid_dataloader)

                valid_loss_summ = tf.summary.Summary()
                valid_loss_summ.value.add(tag='valid_loss', simple_value=float(valid_loss))

                valid_acc_summ = tf.summary.Summary()
                valid_acc_summ.value.add(tag='valid_acc', simple_value=float(valid_acc))

                print("Best valid acc: %.4f, loss: %.4f, acc: %.4f; valid time: %.2fs" % (
                    best_valid_acc, valid_loss, valid_acc, time.time() - start))

                valid_summary_writer.add_summary(valid_loss_summ, step)
                valid_summary_writer.add_summary(valid_acc_summ, step)
                valid_summary_writer.flush()

                # 存模型
                if valid_acc > best_valid_acc:
                    best_valid_acc = valid_acc
                    print("Better model, best_valid_acc: %.4f" % best_valid_acc)
                    save_model(sess, saver, log_root, step)


def trainer(train_dirs,
            valid_dirs,
            init_lr = 0.001,
            batch_size=1024, 
            max_seq_len=100,
            random_scale_factor=0.1, 
            augment_stroke_prob=0.1,

            rnn_size=256, 
            rnn_layers=1, 
            fc_hidden_num = 256,
            input_keep_prob = 1.0,
            output_keep_prob = 1.0,
            state_keep_prob = 1.0,
            fc_keep_prob = 1.0,
            
            log_root="./models/demo"):
            
    print("train data:")
    train_dataloader = CHN_Reco_DataLoader(
            train_dirs,
            is_traing=True,
            batch_size=batch_size,
            max_seq_len=max_seq_len,
            scale_factor=200.0, 
            random_scale_factor=random_scale_factor, 
            augment_stroke_prob=augment_stroke_prob)
    print("valid data:")
    valid_dataloader = CHN_Reco_DataLoader(
            valid_dirs,
            is_traing=False,
            batch_size=batch_size,
            max_seq_len=max_seq_len,
            scale_factor=200.0, 
            random_scale_factor=0.0, 
            augment_stroke_prob=0.0)

    reset_graph()

    model = Content_Enc_Model(                
            init_lr = init_lr,
            batch_size=batch_size, 
            max_seq_len=max_seq_len, 
            class_num=3755, 
            rnn_size=rnn_size, 
            rnn_layers=rnn_layers, 
            fc_hidden_num = fc_hidden_num,
            input_keep_prob = input_keep_prob,
            output_keep_prob = output_keep_prob,
            state_keep_prob = state_keep_prob,
            fc_keep_prob = fc_keep_prob)
    
    valid_model = Content_Enc_Model(                
            init_lr = init_lr,
            batch_size=batch_size, 
            max_seq_len=max_seq_len, 
            class_num=3755, 
            rnn_size=rnn_size, 
            rnn_layers=rnn_layers,
            fc_hidden_num = fc_hidden_num,
            input_keep_prob = 1.0,
            output_keep_prob = 1.0,
            state_keep_prob = 1.0,
            fc_keep_prob = 1.0,
            reuse=True,
            is_training=False)

    sess = tf.InteractiveSession()
    sess.run(tf.global_variables_initializer())

    train(sess, model, valid_model, train_dataloader, valid_dataloader, log_root)


def main():
    train_dirs, valid_dirs, test_dirs = get_data_dirs(os.path.join(OLHWDB_root, "CASIA-OLHWDB1.0"),
                                                        os.path.join(OLHWDB_root, "CASIA-OLHWDB1.1"))
    print(len(train_dirs), len(valid_dirs), len(test_dirs))
    trainer(train_dirs,
            valid_dirs,
            init_lr = 0.001,
            batch_size=1024, 
            max_seq_len=100,
            random_scale_factor=0.1, 
            augment_stroke_prob=0.1,

            rnn_size=512, 
            rnn_layers=1, 
            fc_hidden_num = 512,
            input_keep_prob = 1.0,
            output_keep_prob = 1.0,
            state_keep_prob = 1.0,
            fc_keep_prob = 1.0,
            
            log_root="./models/demo")


if __name__ == '__main__':
    main()
