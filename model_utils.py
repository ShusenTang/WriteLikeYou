import os
import tensorflow as tf


def bi_rnn_encoder(
    inputs, 
    seq_lens,
    layers=1,
    rnn_size=256,
    cell_fn = tf.contrib.rnn.LSTMCell, # tf.contrib.rnn.LayerNormBasicLSTMCell
    input_keep_prob = 1.0,
    output_keep_prob = 1.0,
    state_keep_prob = 1.0,
    scope="bi_rnn_encoder", 
    reuse=tf.AUTO_REUSE):

    with tf.variable_scope(scope, reuse=reuse):
        # TODO
        # 初始化
        output = inputs
        outputs, output_states = None, None
        for i in range(layers):
            with tf.variable_scope("rnn_" + str(i + 1), reuse=tf.AUTO_REUSE):
                cell_fw = tf.nn.rnn_cell.DropoutWrapper( # forward RNNCell
                    cell_fn(num_units=rnn_size),
                    input_keep_prob = input_keep_prob,
                    output_keep_prob = output_keep_prob,
                    state_keep_prob = state_keep_prob)
                cell_bw = tf.nn.rnn_cell.DropoutWrapper( # backward RNNCell
                    cell_fn(num_units=rnn_size),
                    input_keep_prob = input_keep_prob,
                    output_keep_prob = output_keep_prob,
                    state_keep_prob = state_keep_prob)

                outputs, output_states = tf.nn.bidirectional_dynamic_rnn(
                    cell_fw,  # An instance of RNNCell, to be used for forward direction.
                    cell_bw,  # An instance of RNNCell, to be used for backward direction.
                    output,  # input_data
                    sequence_length=seq_lens,
                    swap_memory=True,
                    dtype=tf.float32)

                output = tf.concat(outputs, axis=2)  # output two tensors, tensor shape: (batch_size, max_seq_len, rnn_size)

        output_fw, output_bw = outputs  # tensor shape: ( batch_size, max_seq_len, rnn_size )
        last_h_fw, last_h_bw = output_states
        if isinstance(last_h_fw, tuple):  # lstm: last_h_fw=(c, h), c/h shape: (batch_size, rnn_size)
            last_h_fw = last_h_fw[1]
            last_h_bw = last_h_bw[1]

        all_h = tf.concat([output_fw, output_bw], 2)  # shape: (len(inputs), max_seq_len, 2*rnn_size)
        last_h = tf.concat([last_h_fw, last_h_bw], 1)  # shape: (len(inputs), 2*rnn_size)

        return all_h, last_h


def super_linear(x, output_size, scope=None, reuse=False, init_w='ortho',
                 weight_start=0.0, use_bias=True, bias_start=0.0, input_size=None):
    """Performs linear operation. Uses ortho init defined earlier. """
    shape = x.get_shape().as_list()
    with tf.variable_scope(scope or 'linear'):
        if reuse is True:
            tf.get_variable_scope().reuse_variables()

        w_init = None  # uniform
        if input_size is None:
            x_size = shape[1]
        else:
            x_size = input_size
        if init_w == 'zeros':
            w_init = tf.constant_initializer(0.0)
        elif init_w == 'constant':
            w_init = tf.constant_initializer(weight_start)
        elif init_w == 'gaussian':
            w_init = tf.random_normal_initializer(stddev=weight_start)
        elif init_w == 'ortho':
            w_init = lstm_ortho_initializer(1.0)

        # print(x_size, output_size)
        w = tf.get_variable('super_linear_w', [x_size, output_size], tf.float32, initializer=w_init)
        if use_bias:
            b = tf.get_variable('super_linear_b', [output_size], tf.float32, initializer=tf.constant_initializer(bias_start))
            return tf.matmul(x, w) + b
        return tf.matmul(x, w)

def reset_graph():
    sess = tf.get_default_session()
    if sess:
        sess.close()
    tf.reset_default_graph()

def save_model(sess, saver, model_save_path, global_step):
    checkpoint_path = os.path.join(model_save_path, 'vector')
    print('saving model %s.' % checkpoint_path)
    saver.save(sess, checkpoint_path, global_step=global_step)
    print("Done!")


