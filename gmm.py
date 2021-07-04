import tensorflow as tf
import numpy as np

INF_MIN = 1e-6


def tf_2d_normal(x1, x2, mu1, mu2, s1, s2, rho):
    """Normal distribution"""
    s1 = tf.clip_by_value(s1, 1e-6, 500.0)
    s2 = tf.clip_by_value(s2, 1e-6, 500.0)

    norm1 = tf.subtract(x1, mu1)  # Returns x1-mu1 element-wise
    norm2 = tf.subtract(x2, mu2)
    s1s2 = tf.multiply(s1, s2)

    z = (tf.square(tf.div(norm1, s1)) + tf.square(tf.div(norm2, s2)) -
         2 * tf.div(tf.multiply(rho, tf.multiply(norm1, norm2)), s1s2))
    neg_rho = tf.clip_by_value(1 - tf.square(rho), 1e-6, 1.0)
    result = tf.exp(tf.div(-z, 2 * neg_rho))
    denom = 2 * np.pi * tf.multiply(s1s2, tf.sqrt(neg_rho))
    result = tf.div(result, denom)
    return result


def get_lossfunc(is_training, z_pi, z_mu1, z_mu2, z_sigma1, z_sigma2, z_corr, z_pen,  z_pen_logits, x1_data, x2_data,
                 pen_data, focal_loss_gamma=0):
    result0 = tf_2d_normal(x1_data, x2_data, z_mu1, z_mu2, z_sigma1, z_sigma2,
                           z_corr)
    epsilon = 1e-10
    # result1 is the loss wrt pen offset
    result1 = tf.multiply(result0, z_pi)
    result1 = tf.reduce_sum(result1, 1, keepdims=True)
    result1 = -tf.log(result1 + epsilon)  # avoid log(0)

    fs = 1.0 - pen_data[:, 2]  # use training data for this
    fs = tf.reshape(fs, [-1, 1])
    # Zero out loss terms beyond N_s, the last actual stroke
    result1 = tf.multiply(result1, fs)

    if focal_loss_gamma == 0:
        result2 = tf.nn.softmax_cross_entropy_with_logits_v2(labels=pen_data, logits=z_pen_logits)
    else:
        result2 = focal_loss(pen_data, z_pen, gamma=focal_loss_gamma)

    result2 = tf.reshape(result2, [-1, 1])

    if not is_training:  # eval mode, mask eos columns
        result2 = tf.multiply(result2, fs)
    # result = result1 + result2

    return result1, result2 # result1: pen offset loss, result2: category loss


# below is where we need to do MDN (Mixture Density Network) splitting of distribution params
def get_mixture_coef(output):
    """Returns the tf slices containing mdn dist params."""
    z = output
    z_pen_logits = z[:, 0:3]  # pen states
    z_pi, z_mu1, z_mu2, z_sigma1, z_sigma2, z_corr = tf.split(z[:, 3:], 6, 1)

    # process output z's into MDN paramters

    # softmax all the pi's and pen states:
    z_pi = tf.nn.softmax(z_pi)
    z_pen = tf.nn.softmax(z_pen_logits)

    # exponentiate the sigmas and also make corr between -1 and 1.
    # z_sigma1 = tf.exp(z_sigma1)
    # z_sigma2 = tf.exp(z_sigma2)
    z_sigma1 = tf.minimum(500.0, tf.exp(z_sigma1))
    z_sigma2 = tf.minimum(500.0, tf.exp(z_sigma2))
    z_corr = tf.tanh(z_corr)

    # result = tf.concat([z_pi, z_mu1, z_mu2, z_sigma1, z_sigma2, z_corr, z_pen, z_pen_logits], axis=1)
    result = [z_pi, z_mu1, z_mu2, z_sigma1, z_sigma2, z_corr, z_pen, z_pen_logits]
    return result


def sample_gaussian_2d(mu1, mu2, s1, s2, rho, sqrt_temp=1.0, greedy=False):
    if greedy:
        return mu1, mu2
    mean = [mu1, mu2]
    s1 *= sqrt_temp * sqrt_temp
    s2 *= sqrt_temp * sqrt_temp
    cov = [[s1 * s1, rho * s1 * s2], [rho * s1 * s2, s2 * s2]] 
    x = np.random.multivariate_normal(mean, cov, 1) 
    return x[0][0], x[0][1]


# unused in our paper
def focal_loss(y_true, y_pred, gamma=2):
    """
    focal loss for multi-class classification. fl_loss = -(1 - p_t)^gamma * log(p_t)
    :param y_true: one hot ground truth label. shape: (batch, class_num)
    :param y_pred: prediction (after softmax). shape: (batch, class_num)
    :param gamma:
    :return:
    """
    y_pred += INF_MIN # avoid log(0)

    # cross entropy
    ce = - y_true * tf.log(y_pred)

    # Not necessary to multiply y_true, because weight will be multiplied by ce which has set unconcerned index to 0.
    weight = tf.pow(1 - y_pred, gamma)  
    fl = ce * weight

    reduced_fl = tf.reduce_max(fl, axis=1)  
    return reduced_fl