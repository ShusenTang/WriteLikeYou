import sys

import tensorflow as tf 

sys.path.append("..")
from model_utils import bi_rnn_encoder 


class Content_Enc_Model(object):
    """
    content encoder model (pretrained as a character recognizer), input format-3
    """
    def __init__(self, 
                init_lr = 0.001,
                batch_size=1024, 
                max_seq_len=100, 
                class_num=3755, 
                rnn_size=256, 
                rnn_layers=1, 
                fc_hidden_num = 256,
                input_keep_prob = 1.0,
                output_keep_prob = 1.0,
                state_keep_prob = 1.0,
                fc_keep_prob = 1.0,
                is_training=True,
                reuse=tf.AUTO_REUSE):
        self.init_lr = init_lr
        self.batch_size = batch_size
        self.max_seq_len = max_seq_len
        self.class_num = class_num
        self.rnn_size = rnn_size
        self.rnn_layers = rnn_layers
        self.fc_hidden_num = fc_hidden_num
        self.input_keep_prob = input_keep_prob
        self.output_keep_prob = output_keep_prob
        self.state_keep_prob = state_keep_prob
        self.fc_keep_prob = fc_keep_prob
        self.is_training = is_training

        self.build_model(reuse=reuse)

    def build_model(self, reuse=tf.AUTO_REUSE):
        """Define model architecture."""
        self.input_data = tf.placeholder(dtype=tf.float32, shape=[self.batch_size, self.max_seq_len, 3])
        self.seq_lens = tf.placeholder(dtype=tf.int32, shape=[self.batch_size])
        self.labels = tf.placeholder(dtype=tf.int32, shape=[self.batch_size])

        # encode
        all_h, self.last_h = bi_rnn_encoder(    
                                self.input_data,
                                self.seq_lens,
                                layers=self.rnn_layers,
                                rnn_size=self.rnn_size,
                                cell_fn = tf.contrib.rnn.LSTMCell, 
                                # cell_fn = tf.contrib.rnn.IndyLSTMCell,
                                input_keep_prob = self.input_keep_prob,
                                output_keep_prob = self.output_keep_prob,
                                state_keep_prob = self.state_keep_prob,
                                scope="Content_Encoder", 
                                reuse=reuse)
        
        with tf.variable_scope("classification", reuse=reuse):
            fc_hidden = tf.layers.Dense(self.fc_hidden_num, name='fc_hidden')
            fc_output = tf.layers.Dense(self.class_num, name='fc_output')
            cls_logits = fc_output(tf.layers.dropout(
                                    fc_hidden(self.last_h), 
                                    rate=1.0 - self.fc_keep_prob))

            onehot_labels = tf.one_hot(self.labels, self.class_num)
            self.loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(
                                            labels=onehot_labels, logits=cls_logits))
            self.acc = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(cls_logits, axis=1, output_type=tf.int32),
                                                        self.labels), dtype=tf.float32))

        if self.is_training:
            with tf.variable_scope("optimizer", reuse=reuse):
                self.lr = tf.Variable(self.init_lr, trainable=False)
                optimizer = tf.train.AdamOptimizer(self.lr)
                gvs = optimizer.compute_gradients(self.loss)
                g = 1.0 # grad_clip
                capped_gvs = [(tf.clip_by_value(grad, -g, g), var) for grad, var in gvs if grad is not None]
                self.train_op = optimizer.apply_gradients(capped_gvs)

            with tf.name_scope("summary"):
                loss_summ = tf.summary.scalar("loss", self.loss)
                acc_summ = tf.summary.scalar("accuracy", self.acc)

                self.summ = tf.summary.merge([loss_summ, acc_summ])
        else:
            assert self.input_keep_prob == 1.0
            assert self.output_keep_prob == 1.0
            assert self.state_keep_prob == 1.0
            assert self.fc_keep_prob == 1.0