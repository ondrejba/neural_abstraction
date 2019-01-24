"""
Source: https://github.com/AMLab-Amsterdam/L0_regularization
Paper: https://arxiv.org/abs/1712.01312
I rewrote the code from pytorch to tensorflow.
"""

import os
import numpy as np
import tensorflow as tf

os.environ["CUDA_VISIBLE_DEVICES"] = ""

LIMIT_A = -0.1
LIMIT_B = 1.1
EPSILON = 1e-6

TEMPERATURE = 2 / 3

DROPRATE_INIT = 0.5


def l0_norm(embedding, is_training):

    alpha_shape = [embedding.shape[1].value]
    alpha_loga = tf.get_variable(
        "alpha_loga", shape=None, dtype=tf.float32,
        initializer=tf.random.normal(
            alpha_shape, mean=(np.log(1 - DROPRATE_INIT) - np.log(DROPRATE_INIT)), stddev=1e-2
        )
    )

    z = sample_z(embedding, alpha_loga, is_training)

    sparse_embedding = tf.multiply(embedding, z)

    regularization = reg(alpha_loga)

    return sparse_embedding, regularization


def sample_z(embedding, alpha_loga, is_training):

    # is_training is True:
    eps = get_eps(embedding)
    z_train = quantile_concrete(eps, alpha_loga)
    z_train = hardtanh(z_train, min_val=0, max_val=1)

    # is_training is False:
    z_test = tf.sigmoid(alpha_loga)
    z_test = hardtanh(z_test * (LIMIT_B - LIMIT_A) + LIMIT_A, min_val=0, max_val=1)
    z_test = tf.expand_dims(z_test, axis=0)
    z_test = tf.tile(z_test, [tf.shape(embedding)[0], 1])

    return tf.where(is_training, x=z_train, y=z_test, name="is_training_switch")


def get_eps(embedding):

    return tf.random.uniform((tf.shape(embedding)[0], embedding.shape[1].value), minval=EPSILON, maxval=1 - EPSILON,
                             dtype=tf.float32, name="epsilon")


def quantile_concrete(eps, alpha_loga):

    y = tf.sigmoid((tf.log(eps) - tf.log(1 - eps) + alpha_loga) / TEMPERATURE)
    return y * (LIMIT_B - LIMIT_A) + LIMIT_A


def hardtanh(z, min_val=0, max_val=1):

    return tf.minimum(tf.maximum(z, min_val), max_val)


def cdf_qz(x, alpha_loga):

    xn = (x - LIMIT_A) / (LIMIT_B - LIMIT_A)
    logits = np.log(xn) - np.log(1 - xn)
    return tf.clip_by_value(
        tf.sigmoid(logits * TEMPERATURE - alpha_loga), clip_value_min=EPSILON, clip_value_max=1 - EPSILON
    )


def reg(alpha_loga):

    return tf.reduce_sum(- (1 - cdf_qz(0, alpha_loga)))
