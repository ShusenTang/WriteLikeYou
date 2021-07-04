import tensorflow as tf
import numpy as np


def mask_score(score, mask, score_mask_value, top_k = 0):
    """
    score: [batch, style_input_num*max_seq_len]
    mask:  [batch, style_input_num*max_seq_len], boolean tensor.
    top_k: if top_k > 0, we only take the first top-k value, and the rest are masked
    """
    score_mask_values = score_mask_value * tf.ones_like(score)
    masked_score = tf.where(mask, score, score_mask_values, name="real_len_mask_op")

    if top_k > 0:
        top_k_score, _ = tf.nn.top_k(masked_score, k = top_k) # top_k_score shape: (batch, k)
        top_k_mask = (masked_score >= top_k_score[:, -1:])
        masked_score = tf.where(top_k_mask, masked_score, score_mask_values, name="top_k_mask_op")
    return masked_score



class StyleLuongAttention(tf.contrib.seq2seq.LuongAttention):
    def __init__(self,
                style_input_num,
                num_units, # equal to the dim of query
                memory, # [batch, style_input_num, max_seq_len, hidden_size]
                memory_sequence_length, # [batch, style_input_num]
                scale=False,
                probability_fn=tf.nn.softmax,
                score_mask_value=None,
                top_k = 0,
                dtype=None,
                name="StyleLuongAttention"):
        super(StyleLuongAttention, self).__init__(
                num_units=num_units,
                # memory: [batch, style_input_num*max_seq_len, hidden_size]
                memory=tf.reshape(memory, [memory.shape[0], style_input_num * memory.shape[2], memory.shape[3]]),
                memory_sequence_length=None,
                scale=scale,
                probability_fn=probability_fn,
                score_mask_value=score_mask_value, # The default is -inf
                dtype=dtype,
                name=name)
        
        mask = tf.sequence_mask(memory_sequence_length, memory.shape[2])  # [batch, style_input_num, max_seq_len]
        mask = tf.reshape(mask, [mask.shape[0], -1])  # [batch, style_input_num*max_seq_len]
        if score_mask_value is None:
            score_mask_value = tf.constant(-np.inf)
        if top_k > 0:
            print("When calculating the style attention score, only take the most relevant %d value" % top_k)
        self._probability_fn = lambda score, prev: (probability_fn(
                                                    mask_score(score, mask, score_mask_value, top_k)))


class StyleBahdanauAttention(tf.contrib.seq2seq.BahdanauAttention):
    def __init__(self,
                style_input_num,
                num_units, # Any value, used to unify the dimensions of query and memory
                memory, # [batch, style_input_num, max_seq_len, hidden_size]
                memory_sequence_length, # [batch, style_input_num]
                normalize=False,
                probability_fn=tf.nn.softmax,
                score_mask_value=None,
                top_k = 0,
                dtype=None,
                name="StyleBahdanauAttention"):
        super(StyleBahdanauAttention, self).__init__(
                num_units=num_units,
                # memory: [batch, style_input_num*max_seq_len, hidden_size]
                memory=tf.reshape(memory, [memory.shape[0], style_input_num * memory.shape[2], memory.shape[3]]),
                memory_sequence_length=None,
                normalize=normalize,
                probability_fn=probability_fn,
                score_mask_value=score_mask_value, # The default is -inf
                dtype=dtype,
                name=name)
        mask = tf.sequence_mask(memory_sequence_length, memory.shape[2])  # [batch, style_input_num, max_seq_len]
        mask = tf.reshape(mask, [mask.shape[0], -1])  # [batch, style_input_num*max_seq_len]
        if score_mask_value is None:
            score_mask_value = tf.constant(-np.inf)
        if top_k > 0:
            print("When calculating the style attention score, only take the most relevant %d value" % top_k)
        self._probability_fn = lambda score, prev: (probability_fn(
                                                    mask_score(score, mask, score_mask_value, top_k)))


class StyleNoneAttention(tf.contrib.seq2seq.AttentionMechanism):
    """
    do not use attention: return fixed output
    """
    def __init__(self,
                # style_input_num,
                # num_units,
                memory, # [batch, hidden_size]
                # memory_sequence_length, # [batch, style_input_num]
                dtype=None,
                name="StyleNoneAttention"):

        with tf.name_scope(name):
            self.values = tf.expand_dims(memory, 1) # [batch, 1, hidden_size]
            self.dtype = dtype
            if dtype is None: self.dtype = memory.dtype
            self.batch_size = memory.shape[0]
            self._alignments_size = self.values.shape[1]
        # print(self.values.shape)
        
    def __call__(self, query, state):
        """
        query: shape (batch_size, query_depth)
        state: shape (batch_size, max_time = 1)
        """
        alignments = tf.ones([self.batch_size, 1], dtype=self.dtype)
        next_state = alignments
        return alignments, next_state

    def initial_alignments(self, batch_size, dtype):
        return tf.ones([batch_size, 1], dtype=self.dtype)

    def initial_state(self, batch_size, dtype):
        return self.initial_alignments(batch_size, dtype)

    @property
    def alignments_size(self):
        return self._alignments_size

    @property
    def state_size(self):
        return self._alignments_size


    