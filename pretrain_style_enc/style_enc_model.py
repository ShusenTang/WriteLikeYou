import sys
import numpy as np

import tensorflow as tf 

sys.path.append("..")
from model_utils import bi_rnn_encoder

def AC_loss(embeddings, m = 0.35, s=30, mhe_lambda=0.0):
    """
    L_AC in paper. s*[cos(theta) - m]
    
    Args:
        embeddings: shape of (num_writer, num_entry_per_writer, vec_dim)        
    """
    (num_writer, num_entry_per_writer, vec_dim) = embeddings.get_shape().as_list()
    centroids = tf.reduce_mean(embeddings, axis=1) # (num_writer, vec_dim)

    embeddings_norm = tf.nn.l2_normalize(tf.reshape(embeddings, [-1, vec_dim]), axis=-1)
    centroids_norm = tf.nn.l2_normalize(centroids, axis=-1)

    # cosine similarity matrix, (batch, num_writer)
    cos_theta = tf.matmul(embeddings_norm, centroids_norm, transpose_b=True)
    # eps = 1e-10
    # cos_theta = tf.clip_by_value(cos_theta, -1+eps, 1-eps)
    phi = cos_theta - m # (m / (1.0 + anneal_lambda))

    # [i for i in range(3) for _ in range(2)] = [0, 0, 1, 1, 2, 2]
    labels = tf.constant([w for w in range(num_writer) for _ in range(num_entry_per_writer)])
    label_onehot = tf.one_hot(labels, num_writer)
    logits = s * tf.where(tf.equal(label_onehot, 1), phi, cos_theta)

    batch_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=labels, logits=logits)
    ac_loss = tf.reduce_mean(batch_loss)

    if mhe_lambda > 0.0:
        mhe_loss = MHE_loss(centroids_norm)
    else:
        mhe_loss = tf.constant(0.0)

    total_loss = ac_loss + mhe_lambda * mhe_loss

    return cos_theta, ac_loss, mhe_loss, total_loss


def MHE_loss(normed_w):
    """
    1 / ||w1 - w2||^2
    normed_w: (class_num, dim), normed_w = tf.nn.l2_normalize(w, axis=-1)
    """
    class_num = normed_w.get_shape().as_list()[0]

    # ||w1 - w2||^2 = ||w1||^2 - 2w1w2 + ||w2||^2 = 2 - 2w1w2
    dist = 2.0 - 2.0 * tf.matmul(normed_w, tf.transpose(normed_w)) # (class_num, class_num)

    mask = tf.eye(class_num) # identity matrix of shape (class_num, class_num)

    # masked_dist = (1.0 - mask) * dist + mask * 1e6
    mhe_loss_matrix = (1.0 - mask) / (dist + mask) #  (dist + mask) is avoid dividing by zero
    mhe_loss = tf.reduce_sum(mhe_loss_matrix) / float(class_num * (class_num - 1))

    return mhe_loss


class Style_Enc_Model(object):
    """
    style encoder (pretrained as a style classifier), input format-3
    """
    def __init__(self, 
                init_lr = 0.001,
                num_writer_per_batch=64,
                num_entry_per_writer=16,
                max_seq_len=110, 
                rnn_size=256, 
                rnn_layers=1, 
                embedding_dim = 256,
                ac_softmax_m = 0.35, 
                ac_softmax_s = 30,
                mhe_lambda = 0.0,
                input_keep_prob = 1.0,
                output_keep_prob = 1.0,
                state_keep_prob = 1.0,
                is_training=True,
                reuse=tf.AUTO_REUSE):
        self.init_lr = init_lr
        self.num_writer_per_batch = num_writer_per_batch
        self.num_entry_per_writer = num_entry_per_writer
        self.batch_size = num_writer_per_batch * num_entry_per_writer
        self.max_seq_len = max_seq_len
        self.rnn_size = rnn_size
        self.rnn_layers = rnn_layers
        self.embedding_dim = embedding_dim
        self.ac_softmax_m = ac_softmax_m
        self.ac_softmax_s = ac_softmax_s
        self.mhe_lambda = mhe_lambda
        self.input_keep_prob = input_keep_prob
        self.output_keep_prob = output_keep_prob
        self.state_keep_prob = state_keep_prob
        self.is_training = is_training

        self.build_model(reuse=reuse)

    def build_model(self, reuse=tf.AUTO_REUSE):
        """Define model architecture."""
        self.input_data = tf.placeholder(dtype=tf.float32, shape=[self.batch_size, self.max_seq_len, 3])
        self.seq_lens = tf.placeholder(dtype=tf.int32, shape=[self.batch_size])

        # encode
        all_h, last_h = bi_rnn_encoder(    
                                self.input_data,
                                self.seq_lens,
                                layers=self.rnn_layers,
                                rnn_size=self.rnn_size,
                                cell_fn = tf.contrib.rnn.LSTMCell, 
                                input_keep_prob = self.input_keep_prob,
                                output_keep_prob = self.output_keep_prob,
                                state_keep_prob = self.state_keep_prob,
                                scope="Style_Encoder", 
                                reuse=reuse)

        # attention
        with tf.variable_scope("attention", reuse=reuse):
            mask = tf.expand_dims(tf.sequence_mask(self.seq_lens, maxlen = self.max_seq_len), axis=-1) # bool, [batch, max_seq_len, 1]
            all_h_mean = tf.reduce_mean(all_h * tf.cast(mask, dtype=tf.float32), axis=1) # (batch, 2*rnn_size)
            attn_linear = tf.layers.Dense(last_h.get_shape()[-1], name='attn_linear_layer')
            query = attn_linear(all_h_mean) # (batch, 2*rnn_size)
            scores = tf.matmul(all_h, tf.expand_dims(query, -1)) # (batch, max_seq_len, 1)
            masked_scores = tf.where(mask, scores, -np.inf*tf.ones_like(scores)) # (batch, max_seq_len, 1)
            
            masked_weight = tf.nn.softmax(masked_scores, axis=1) # (batch, max_seq_len, 1)
            self.weighted_all_h = tf.reduce_sum(all_h * masked_weight, axis=1)  # (batch, 2*rnn_size)
        
        with tf.variable_scope("classification", reuse=reuse):
            linear_layer = tf.layers.Dense(self.embedding_dim, name='linear_layer')
            self.embeddings = linear_layer(self.weighted_all_h) # (batch_size, embedding_dim)

            self.cos_sim, self.ac_loss, self.mhe_loss, self.loss = AC_loss(
                                                                        tf.reshape(self.embeddings, [self.num_writer_per_batch, 
                                                                                                     self.num_entry_per_writer, 
                                                                                                     self.embedding_dim]),
                                                                        m=self.ac_softmax_m, 
                                                                        s=self.ac_softmax_s,
                                                                        mhe_lambda=self.mhe_lambda)

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
                ac_loss_summ = tf.summary.scalar("ac_loss", self.ac_loss)
                mhe_loss_summ = tf.summary.scalar("mhe_loss", self.mhe_loss)
                cos_sim_sum = tf.summary.image("cos_sim", 
                                               tf.reshape(self.cos_sim,[1, self.batch_size, self.num_writer_per_batch, 1]),
                                               max_outputs=1)

                self.summ = tf.summary.merge([loss_summ, ac_loss_summ, mhe_loss_summ, cos_sim_sum])
        else:
            assert self.input_keep_prob == 1.0
            assert self.output_keep_prob == 1.0
            assert self.state_keep_prob == 1.0
