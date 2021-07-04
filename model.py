# -*- coding: utf-8 -*-

import os
import time
import tensorflow as tf
import numpy as np

import gmm
from style_attention import StyleLuongAttention, StyleBahdanauAttention, StyleNoneAttention

from model_utils import bi_rnn_encoder, super_linear


def copy_hparams(hparams):
    """Return a copy of an HParams instance."""
    return tf.contrib.training.HParams(**hparams.values())

# CUDA_VISIBLE_DEVICES=8 python train.py --hparams="log_root=./models/1227_enc2layer4_dec512, dec_rnn_size=512"
def get_default_hparams():
    hparams = tf.contrib.training.HParams(
        # data_set=['FZJLJW_775.npz'],  # Our dataset.
        # data_dir="./datasets/mean_4.0",  # The directory in which to find the datasets.
        # data_root="/media/tss/S1/hanzi/npy_relative_dist/rdp4.0",
        # ref_dir="mean",
        log_root="./models/debug",  # Directory to store model checkpoints, tensorboard.
        pretrain_content_enc="./pretrain_content_enc/models/926_LSTM",
        pretrain_style_enc="./pretrain_style_enc/models/1224_layer4",

        num_steps=1000000,  # Total number of steps of training. Keep large.
        save_every=2500,  # Number of batches per checkpoint creation.
        max_seq_len=110,  

        # Bi-RNN Encoder for content encoding
        enc1_rnn_size=256,
        enc1_rnn_layers=1,
        enc1_model='lstm', # lstm or gru, recommend lstm

        # Bi-RNN Encoder for style encoding
        enc2_rnn_size=256,
        enc2_rnn_layers=4,
        enc2_model='lstm',
        
        style_input_num=10,
        # style_embed_size=128,
        # style_pool_mode="max", # or mean
        style_attention_method='L', # L(Luong) or B(bahdanau) or N(none)
        style_attention_topk=0,

        dec_rnn_size=512,
        dec_rnn_layers=1,
        dec_model='lstm',

        batch_size=128,
        grad_clip=1.0,  # Gradient clipping. Recommend leaving at 1.0.
        # max_grad_norm=1.0,

        num_mixture=20,  # Number of mixtures in Gaussian mixture model.
        learning_rate=0.001, # Learning rate.
        decay_rate=0.9999,  # Learning rate decay per minibatch.
        min_learning_rate=0.000001,  # Minimum learning rate.

        # input dropout, output dropout, recurrent dropout 分析
        # use_input_dropout=False,   # Input dropout. Recommend leaving False.
        input_keep_prob=1.0,   # Probability of input dropout keep.
        # use_output_dropout=False,  # Output droput. Recommend leaving False.
        output_keep_prob=1.0,  # Probabilpity of output dropout keep.
        state_keep_prob=1.0, 

        random_scale_factor=0.10,  # Random scaling data augmention proportion.
        augment_stroke_prob=0.10,  # Point dropping augmentation proportion.
        scale_factor=200.0, # scale the pen offset manually, keep 200.

        is_training=True,  # Is model training
        is_testing=False,  # Is model testing
        attention_method='LM',  # L: LuongAttention, B: BahdanauAttention, LM: LuongMonotonicAttention. Recommend LM.

        fl_gamma=0,  # focal loss gamma, keep 0 in paper
        lp_weight=2.0, 

        style_cycle_weight=0.0, 
        content_cycle_weight=0.0, 
        cycle_loss_decay=0.99996
        # in_class_loss_weight=1.0
    )
    return hparams


def decoder(hps, dec_input, enc_last_h, enc_all_h, enc_seq_lens, 
            style_enc_all_h, style_enc_seq_lens, style_enc_last_h,
            input_keep_prob, output_keep_prob, state_keep_prob,
            scope="Decoder", reuse=tf.AUTO_REUSE):
    """
    style_enc_all_h: [batch, style_input_num, max_seq_len, size]
    style_enc_last_h: [batch, style_input_num, size]
    """
    with tf.variable_scope(scope, reuse=reuse):
        if hps.dec_model == "gru":
            dec_cell_fn = tf.nn.rnn_cell.GRUCell
        elif hps.dec_model == "lstm":
            dec_cell_fn = tf.contrib.rnn.LSTMCell
        else:
            raise ValueError("dec_model should be gru or lstm")


        dec_cell = tf.nn.rnn_cell.DropoutWrapper(
                    dec_cell_fn(num_units=hps.dec_rnn_size),
                    input_keep_prob = input_keep_prob,
                    output_keep_prob = output_keep_prob,
                    state_keep_prob = state_keep_prob)

        if hps.dec_rnn_layers > 1:
            cell_list = [tf.nn.rnn_cell.DropoutWrapper(
                                    dec_cell_fn(num_units=hps.dec_rnn_size),
                                    input_keep_prob = input_keep_prob,
                                    output_keep_prob = output_keep_prob,
                                    state_keep_prob = state_keep_prob)
                         for i in range(hps.dec_rnn_layers)]
            dec_cell = tf.nn.rnn_cell.MultiRNNCell(cell_list)


        init_state_output_size = hps.dec_rnn_size
        if "lstm" in hps.dec_model:
            init_state_output_size *= 2  # h anc c are all (batch_size x dec_rnn_size)

        initial_state = tf.nn.tanh(  # [h0; C0] = tanh(Wz*Z + bz) 
            super_linear(
                enc_last_h,
                output_size=init_state_output_size,
                init_w='gaussian',
                weight_start=0.001,
                input_size=hps.enc1_rnn_size * 2))  # shape = (batch_size, hps.dec_rnn_size * 2)
        if "lstm" in hps.dec_model:
            c, h = tf.split(initial_state, 2, 1)
            decoder_initial_state = tf.contrib.rnn.LSTMStateTuple(c, h)
        else:
            decoder_initial_state = initial_state

        if hps.dec_rnn_layers > 1:
            states_list = []
            for i in range(hps.dec_rnn_layers):
                states_list.append(decoder_initial_state)
            decoder_initial_state = tuple(states_list)

        # three function CustomHelper needed
        def initial_fn():
            initial_elements_finished = (1 < np.zeros(hps.batch_size))  # false, shape:(batch_size)
            initial_input = np.zeros((hps.batch_size, 5), dtype=np.float32)
            initial_input[:, 2] = 1  
            return initial_elements_finished, initial_input

        def sample_fn(time, outputs, state): 
            unused_sample_ids = tf.zeros([outputs.shape[0]])
            return unused_sample_ids

        def next_inputs_fn(time, outputs, state, sample_ids): 
            [pi, mu1, mu2, sigma1, sigma2, corr, pen, pen_logits] = gmm.get_mixture_coef(outputs)

            idx_eos = tf.argmax(pen, axis=1)
            eos = tf.one_hot(idx_eos, depth=3)

            # -------------- Weighted average of all mixtures -----------------
            # next_x1 = tf.reduce_sum(tf.multiply(mu1, pi), axis=1)
            # next_x2 = tf.reduce_sum(tf.multiply(mu2, pi), axis=1)
            
            # -------------- Take only the mixture with the largest weight -----------------
            max_mixture_idx = tf.stack([tf.range(pi.shape[0], dtype=tf.int64), tf.argmax(pi, axis=1)], axis=1)
            next_x1 = tf.gather_nd(mu1, max_mixture_idx)
            next_x2 = tf.gather_nd(mu2, max_mixture_idx)

            next_x = tf.stack([next_x1, next_x2], axis=1)
            next_input = tf.concat([next_x, eos], axis=1)

            # print(next_inputs.shape)  # (batch_size, 5)

            tmp = tf.ones([next_x.shape[0]])
            elements_finished_1 = tf.equal(tmp, eos[:, -1])  # this operation produces boolean tensor of [batch_size]
            elements_finished_2 = (time >= hps.max_seq_len)  # this operation produces boolean tensor of [batch_size]

            elements_finished = tf.logical_or(elements_finished_1, elements_finished_2)
            # all_finished = tf.reduce_all(elements_finished)  # -> boolean scalar
            # next_inputs = tf.cond(all_finished, lambda: pad_step_embedded, lambda: next_inputs)
            next_state = state
            return elements_finished, next_input, next_state

        my_inference_helper = tf.contrib.seq2seq.CustomHelper(initial_fn, sample_fn, next_inputs_fn)

        # Create an attention mechanism
        assert hps.attention_method in ['L', 'B', 'LM', 'BM']
        if hps.attention_method == 'L':
            tf.logging.info('Content encoder using LuongAttention.')
            attention_mechanism = tf.contrib.seq2seq.LuongAttention(
                                                        # The depth of the attention mechanism. 
                                                        num_units=hps.dec_rnn_size, # equal to the dim of query
                                                        memory=enc_all_h,  # self.enc_all_h,
                                                        memory_sequence_length=enc_seq_lens,
                                                        name='LuongAttention'
                                                    )
        elif hps.attention_method == 'B':
            tf.logging.info('Content encoder using BahdanauAttention.')
            attention_mechanism = tf.contrib.seq2seq.BahdanauAttention(
                                                        # The depth of the query mechanism.
                                                        num_units=hps.dec_rnn_size, # Any value, used to unify the dim of query and memory
                                                        memory=enc_all_h,  # self.enc_all_h,
                                                        memory_sequence_length=enc_seq_lens,
                                                        name='BahdanauAttention'
                                                    )
        elif hps.attention_method == 'LM':
            tf.logging.info('Content encoder using LuongMonotonicAttention.')
            attention_mechanism = tf.contrib.seq2seq.LuongMonotonicAttention(
                                                        # The depth of the query mechanism.
                                                        num_units=hps.dec_rnn_size,
                                                        memory=enc_all_h,  # self.enc_all_h,
                                                        memory_sequence_length=enc_seq_lens,
                                                        # mode='hard',
                                                        name='LuongMonotonicAttention'
                                                    )
        elif hps.attention_method == 'BM':
            tf.logging.info('Content encoder using BahdanauMonotonicAttention.')
            attention_mechanism = tf.contrib.seq2seq.BahdanauMonotonicAttention(
                                                        # The depth of the query mechanism.
                                                        num_units=hps.dec_rnn_size,
                                                        memory=enc_all_h,  # self.enc_all_h,
                                                        memory_sequence_length=enc_seq_lens,
                                                        normalize=True,
                                                        name='BahdanauMonotonicAttention'
                                                    )
        else:
            raise ValueError("attention_method must be L, B, LM or BM, not %s" % hps.attention_method)


        # ------------------------------------- attention for style encoder -----------------------------
        if hps.style_attention_method == 'L':
            tf.logging.info('Style encoder using LuongAttention.')
            style_attention_mechanism = StyleLuongAttention(
                                                        style_input_num=hps.style_input_num,
                                                        num_units=hps.dec_rnn_size, # equal to the dim of query
                                                        memory=style_enc_all_h,
                                                        memory_sequence_length=style_enc_seq_lens,
                                                        top_k=hps.style_attention_topk,
                                                        name='StyleLuongAttention'
                                                    )
        elif hps.style_attention_method == 'B':
            tf.logging.info('Style encoder using BahdanauAttention.')
            style_attention_mechanism = StyleBahdanauAttention(
                                                        style_input_num=hps.style_input_num,
                                                        num_units=128, # # Any value, used to unify the dim of query and memory
                                                        memory=style_enc_all_h, # [batch, style_input_num, max_seq_len, size]
                                                        memory_sequence_length=style_enc_seq_lens, # [batch, style_input_num]
                                                        top_k = hps.style_attention_topk,
                                                        name='StyleBahdanauAttention'
                                                    )
        elif hps.style_attention_method == 'N':
            tf.logging.info("Style encoder do NOT use attention!!!")
            style_attention_mechanism = StyleNoneAttention(
                                                        memory=tf.reduce_mean(style_enc_last_h, 1), # [batch, size]
                                                        name='StyleNoneAttention'
                                                    )
        else:
            raise ValueError("Style_attention_method must be L, B or N, not %s" % hps.style_attention_method)


        dec_cell_with_attn = tf.contrib.seq2seq.AttentionWrapper(
                                        dec_cell,
                                        [attention_mechanism, style_attention_mechanism],
                                        attention_layer_size=[hps.dec_rnn_size, hps.dec_rnn_size],  
                                        alignment_history=True, 
                                        name='AttentionWrapper'
                                        )

        # Helper
        if hps.is_testing: # test
            helper = my_inference_helper
        else: # train or valid
            helper = tf.contrib.seq2seq.TrainingHelper(inputs=dec_input,
                                                       sequence_length=tf.convert_to_tensor([hps.max_seq_len for _ in range(hps.batch_size)]),
                                                       name='TrainingHelper')

        # Decoder
        att_wrapper_initial_state = dec_cell_with_attn.zero_state(batch_size=hps.batch_size, dtype=tf.float32).clone(
                                                                                     cell_state=decoder_initial_state)
        n_out = (3 + hps.num_mixture * 6)  # decoder output dim
        fc_layer = tf.layers.Dense(n_out, name='output_fc')
        decoder = tf.contrib.seq2seq.BasicDecoder(dec_cell_with_attn, helper,
                                                  initial_state=att_wrapper_initial_state,
                                                  output_layer=fc_layer)
        # Dynamic decoding
        decoder_final_outputs, decoder_final_state, final_sequence_lengths = tf.contrib.seq2seq.dynamic_decode(
                                                  decoder=decoder,
                                                  scope='dynamic_decode')

        # print(decoder_final_state)
        output, sample_id = decoder_final_outputs # this is a AttentionWrapperState
        cell_state, att, time, alignments, alignment_history, attention_state = decoder_final_state

        # alignment_history is a tensorArray 
        timemajor_alignment_history = alignment_history[0].stack() # tensor
        style_timemajor_alignment_history = alignment_history[1].stack()

        return output, timemajor_alignment_history, style_timemajor_alignment_history


class Model(object):
    def __init__(self, hps, reuse=tf.AUTO_REUSE):
        self.hps = hps
        self.build_model(reuse=reuse)

    def build_model(self, reuse=tf.AUTO_REUSE):
        """Define model architecture."""
        self.enc1_seq_lens = tf.placeholder(dtype=tf.int32, shape=[self.hps.batch_size])  # actual length
        self.enc2_seq_lens = tf.placeholder(dtype=tf.int32, shape=[self.hps.batch_size * self.hps.style_input_num])  
        self.dec_seq_lens = tf.placeholder(dtype=tf.int32, shape=[self.hps.batch_size])  

        self.enc1_input_data = tf.placeholder(
            dtype=tf.float32, shape=[self.hps.batch_size, self.hps.max_seq_len, 3]) # format-3
        self.enc2_input_data = tf.placeholder(
            dtype=tf.float32, shape=[self.hps.batch_size*self.hps.style_input_num, self.hps.max_seq_len, 3]) # format-3
        self.dec_input_data = tf.placeholder(
            dtype=tf.float32, shape=[self.hps.batch_size, self.hps.max_seq_len + 1, 5])

        # encode1
        enc1_all_h, enc1_last_h = bi_rnn_encoder(
                                        inputs = self.enc1_input_data, 
                                        seq_lens = self.enc1_seq_lens,
                                        layers=self.hps.enc1_rnn_layers,
                                        rnn_size=self.hps.enc1_rnn_size,
                                        cell_fn = tf.contrib.rnn.LSTMCell,
                                        input_keep_prob = self.hps.input_keep_prob,
                                        output_keep_prob = self.hps.output_keep_prob,
                                        state_keep_prob = self.hps.state_keep_prob,
                                        scope="Content_Encoder", 
                                        reuse=reuse)
        # encode2
        enc2_all_h, enc2_last_h = bi_rnn_encoder(
                                        inputs = self.enc2_input_data, 
                                        seq_lens = self.enc2_seq_lens,
                                        layers=self.hps.enc2_rnn_layers,
                                        rnn_size=self.hps.enc2_rnn_size,
                                        cell_fn = tf.contrib.rnn.LSTMCell,
                                        input_keep_prob = self.hps.input_keep_prob,
                                        output_keep_prob = self.hps.output_keep_prob,
                                        state_keep_prob = self.hps.state_keep_prob,
                                        scope="Style_Encoder", 
                                        reuse=reuse)

        reshaped_enc2_last_h = tf.reshape(enc2_last_h, [self.hps.batch_size, self.hps.style_input_num, -1])

        mean_enc2_last_h = tf.reduce_mean(reshaped_enc2_last_h, axis=1, keepdims=True) # shape: (batch_size, 1, h_dim)

        # decode
        reshaped_enc2_all_h = tf.reshape(enc2_all_h, [self.hps.batch_size, self.hps.style_input_num, self.hps.max_seq_len, -1])
        reshaped_enc2_seq_lens = tf.reshape(self.enc2_seq_lens, [self.hps.batch_size, self.hps.style_input_num])
        dec_input = self.dec_input_data[:, :self.hps.max_seq_len, :]  # S0~S(max-1)
        # dec_actual_input = tf.concat([dec_input, overlay_style_embed], axis=-1)  
        dec_out, self.timemajor_alignment_history, self.style_timemajor_alignment_history = \
                                                    decoder(self.hps, dec_input, enc1_last_h, enc1_all_h, self.enc1_seq_lens,
                                                            reshaped_enc2_all_h, reshaped_enc2_seq_lens, reshaped_enc2_last_h,
                                                            input_keep_prob = self.hps.input_keep_prob,
                                                            output_keep_prob = self.hps.output_keep_prob,
                                                            state_keep_prob = self.hps.state_keep_prob,
                                                            scope="Decoder", reuse=reuse)


        n_out = (3 + self.hps.num_mixture * 6)  # decoder output dim
        dec_out = tf.reshape(dec_out, [-1, n_out])  # shape = (batch_size * max_seq_len, n_out)
        # self.final_state = tf.concat([c, h], axis=1)

        # the shape of first six tensors: (batch_size * max_seq_le, num_mixture), last two: (batch_size * max_seq_le, 3)
        [o_pi, o_mu1, o_mu2, o_sigma1, o_sigma2, o_corr, o_pen, o_pen_logits] = gmm.get_mixture_coef(dec_out)
        # reshape target data so that it is compatible with prediction shape
        target = tf.reshape(self.dec_input_data[:, 1:self.hps.max_seq_len + 1, :], [-1, 5])  # (batch_size * max_seq_le, 5)
        [x1_data, x2_data, eos_data, eoc_data, cont_data] = tf.split(target, 5, 1)
        pen_data = tf.concat([eos_data, eoc_data, cont_data], 1)

        # Ls for pen offset, Lp for category
        Ls, Lp = gmm.get_lossfunc(self.hps.is_training, o_pi, o_mu1, o_mu2, o_sigma1, o_sigma2,
                                      o_corr, o_pen, o_pen_logits, x1_data, x2_data, pen_data, self.hps.fl_gamma)

        # self.Ls = tf.reduce_mean(Ls)
        self.Ls = tf.reduce_sum(Ls) / tf.to_float(tf.reduce_sum(self.dec_seq_lens))
        self.Lp = tf.reduce_mean(Lp)

        # ************************************ cycle loss **************************************
        if self.hps.content_cycle_weight > 0.0 or self.hps.style_cycle_weight > 0.0:
            fake_delta_x = tf.reduce_sum(tf.multiply(o_pi, o_mu1), 1, keepdims=True) # (batch_size * max_seq_le, 1)
            fake_delta_y = tf.reduce_sum(tf.multiply(o_pi, o_mu2), 1, keepdims=True)
            # fake_delta_xy = tf.reshape(tf.concat([fake_delta_x, fake_delta_y], axis=1), shape=[self.hps.batch_size, self.hps.max_seq_len, 2])
            fake_flatten_p2 = tf.where(tf.greater(o_pen[:, 1:2], o_pen[:, 0:1]), # (p2 > p1)
                                    tf.ones(shape=(self.hps.batch_size * self.hps.max_seq_len, 1), dtype=tf.float32),
                                    tf.zeros(shape=(self.hps.batch_size * self.hps.max_seq_len, 1), dtype=tf.float32))
            fake_data = tf.reshape(
                tf.concat([fake_delta_x, fake_delta_y, fake_flatten_p2], axis=1), # (batch_size * max_seq_le, 3)
                shape = [self.hps.batch_size, self.hps.max_seq_len, 3]
            )
        # *************** 1. content cycle loss *********
        if self.hps.content_cycle_weight > 0.0:
            fake_enc1_all_h, fake_enc1_last_h = bi_rnn_encoder(
                                                    inputs = fake_data, 
                                                    seq_lens = self.dec_seq_lens,
                                                    layers=self.hps.enc1_rnn_layers,
                                                    rnn_size=self.hps.enc1_rnn_size,
                                                    cell_fn = tf.contrib.rnn.LSTMCell,
                                                    input_keep_prob = self.hps.input_keep_prob,
                                                    output_keep_prob = self.hps.output_keep_prob,
                                                    state_keep_prob = self.hps.state_keep_prob,
                                                    scope="Content_Encoder", reuse=True)
            self.content_cycle_loss = tf.losses.absolute_difference(
                                            enc1_last_h,
                                            fake_enc1_last_h, weights=1.0)
        else:
            self.content_cycle_loss = tf.constant(0.0)
        # ***************** 2. style cycle loss *********
        if self.hps.style_cycle_weight > 0.0:
            fake_enc2_all_h, fake_enc2_last_h = bi_rnn_encoder(
                                                    inputs = fake_data, 
                                                    seq_lens = self.dec_seq_lens,
                                                    layers=self.hps.enc2_rnn_layers,
                                                    rnn_size=self.hps.enc2_rnn_size,
                                                    cell_fn = tf.contrib.rnn.LSTMCell,
                                                    input_keep_prob = self.hps.input_keep_prob,
                                                    output_keep_prob = self.hps.output_keep_prob,
                                                    state_keep_prob = self.hps.state_keep_prob,
                                                    scope="Style_Encoder", reuse=True)
            self.style_cycle_loss = tf.losses.absolute_difference(
                                            tf.reshape(mean_enc2_last_h, shape=fake_enc2_last_h.shape), 
                                            fake_enc2_last_h, weights=1.0)
        else:
            self.style_cycle_loss = tf.constant(0.0)
        # ********************************************************************************************


        self.pi = o_pi
        self.mu1 = o_mu1
        self.mu2 = o_mu2
        self.sigma1 = o_sigma1
        self.sigma2 = o_sigma2
        self.corr = o_corr
        self.pen_logits = o_pen_logits
        self.pen = o_pen

        self.r_cost = self.Ls + self.hps.lp_weight * self.Lp
        
        with tf.variable_scope("curr_cycle_loss_weight", reuse=reuse):
            self.curr_cycle_loss_weight = tf.Variable(0.0, trainable=False)

        self.cost = self.r_cost  + \
                    self.curr_cycle_loss_weight * self.hps.content_cycle_weight * self.content_cycle_loss + \
                    self.curr_cycle_loss_weight * self.hps.style_cycle_weight * self.style_cycle_loss
        
        if self.hps.is_training:
            with tf.variable_scope("optimizer", reuse=reuse):
                with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)): # for BN
                    self.lr = tf.Variable(self.hps.learning_rate, trainable=False)
                    optimizer = tf.train.AdamOptimizer(self.lr)
                    
                    tvars = tf.trainable_variables()
                    if self.hps.pretrain_content_enc != "":
                        print("Use pretrained content encoder %s (freeze Content_Encoder)." % self.hps.pretrain_content_enc)
                        tvars = [var for var in tvars if 'Content_Encoder' not in var.name]
                    if self.hps.pretrain_style_enc != "":
                        print("Use pretrained style encoder %s (freeze Style_Encoder)." % self.hps.pretrain_style_enc)
                        tvars = [var for var in tvars if 'Style_Encoder' not in var.name]
                    
                    print("\nTraining variables:")
                    for var in tvars:
                        print(var.name)
                    gvs = optimizer.compute_gradients(self.cost, var_list=tvars)
                    g = self.hps.grad_clip

                    capped_gvs = [(tf.clip_by_value(grad, -g, g), var) for grad, var in gvs if grad is not None]
                    self.train_op = optimizer.apply_gradients(capped_gvs)

            with tf.name_scope("summary"):
                lr_summ = tf.summary.scalar("lr", self.lr)
                cost_summ = tf.summary.scalar("cost", self.cost)
                r_cost_summ = tf.summary.scalar("r_cost", self.r_cost)
                Ls_loss_summ = tf.summary.scalar("Ls_loss", self.Ls)
                Lp_loss_summ = tf.summary.scalar("Lp_loss", self.Lp)
                cycle_loss_weight_summ = tf.summary.scalar("cycle_loss_weight", self.curr_cycle_loss_weight)
                content_cycle_loss_summ = tf.summary.scalar("content_cycle_loss", self.content_cycle_loss)
                style_cycle_loss_summ = tf.summary.scalar("style_cycle_loss", self.style_cycle_loss)

                self.summ = tf.summary.merge([lr_summ, cost_summ, r_cost_summ, Ls_loss_summ, Lp_loss_summ,
                                              cycle_loss_weight_summ, content_cycle_loss_summ, style_cycle_loss_summ])
        else:
            assert self.hps.input_keep_prob == 1.0
            assert self.hps.output_keep_prob == 1.0
            assert self.hps.state_keep_prob == 1.0
