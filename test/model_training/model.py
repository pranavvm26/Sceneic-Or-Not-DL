"""
Build the Inception v3 network
The Inception v3 architecture is described in http://arxiv.org/abs/1512.00567
"""

import re
import tensorflow as tf
from tensorflow.contrib.slim.nets import inception

FLAGS = tf.app.flags.FLAGS
TOWER_NAME = 'tower'
LOSSES_COLLECTION = '_losses'
LOGIT_NAMES = ['InceptionV3/Logits', 'InceptionV3/AuxLogits']


def inference(images, dropout_rate, is_training=True, scope='InceptionV3'):
    """
    Build Inception v3 model architecture.

    :param images: 4-D Tensor of images
    :param num_classes: Number of classes
    :param dropout_rate: Keep rate for dropout layers in training models
    :param is_training: If set to `True`, build the inference model for training.
        Kernels that operate differently for inference during training
        e.g. dropout, are appropriately configured.
    :param scope: Optional prefix string identifying the tower.
    :return: List [2-D Tensor, 2-D Tensor]
             Logits. 2-D float Tensor.
             Auxiliary Logits. 2-D float Tensor of side-head. Used for training only.
    """

    logits, endpoints = inception.inception_v3(inputs=images,
                                               num_classes=1,
                                               is_training=is_training,
                                               dropout_keep_prob=dropout_rate,
                                               scope=scope)

    # Add summaries for viewing model statistics on TensorBoard.
    _activation_summaries(endpoints)

    # Grab the logits associated with the side head. Employed during training.
    auxiliary_logits = endpoints['AuxLogits']

    return [tf.squeeze(logits, -1), tf.squeeze(auxiliary_logits, -1)]


def loss(logits, labels):
    """
    Adds all losses for the model.

    Note the final loss is not returned. Instead, the list of losses are collected
    by slim.losses. The losses are accumulated in tower_loss() and summed to
    calculate the total loss.

    :param logits: Network output logits
    :param labels: Dense label seC
    :param num_classes: Number of classes to encode
    """
    # shape
    print("Logit Shape: {0}, Aux Logit Shape: {1}, Label Shape: {2}".format(logits[0].get_shape(),
                                                                            logits[1].get_shape(),
                                                                            labels.get_shape()))

    # Cross entropy loss for the main softmax prediction.
    _cross_entropy_loss(logits[0], logits[1], labels, scope='total-loss')

#
# def _cross_entropy_loss(logits, auxlogits, labels, weight=1.0, scope=None):
#
#     logits.get_shape().assert_is_compatible_with(labels.get_shape())
#     with tf.name_scope(scope, 'CrossEntropyLoss', [logits, labels]):
#
#         mse = tf.losses.mean_squared_error(logits, labels)
#
#         weight = tf.convert_to_tensor(weight,
#                                       dtype=logits.dtype.base_dtype,
#                                       name='loss_weight')
#         loss = tf.multiply(weight, tf.reduce_mean(mse), name='value')
#
#         tf.add_to_collection(LOSSES_COLLECTION, loss)
#
#         return loss


def _cross_entropy_loss(logits, auxlogits, labels, scope=None):
    logits.get_shape().assert_is_compatible_with(labels.get_shape())
    with tf.name_scope(scope, 'CrossEntropyLoss', [logits, labels]):
        mse_logit = tf.losses.mean_squared_error(logits, labels)
        mse_auxlogit = tf.losses.mean_squared_error(auxlogits, labels)

        loss_logit = tf.multiply(1.0, tf.reduce_mean(mse_logit), name='value')
        loss_auxlogit = tf.multiply(0.4, tf.reduce_mean(mse_auxlogit), name='aux-value')

        loss = loss_logit + loss_auxlogit

        tf.add_to_collection(LOSSES_COLLECTION, loss)

        return loss


def _activation_summary(x):
    """
    Helper to create summaries for activations.
    Creates a summary that provides a histogram of activations.
    Creates a summary that measure the sparsity of activations.

    :param x: Tensor
    """

    tensor_name = re.sub('%s_[0-9]*/' % TOWER_NAME, '', x.op.name)
    tf.summary.histogram(tensor_name + '/activations', x)
    tf.summary.scalar(tensor_name + '/sparsity', tf.nn.zero_fraction(x))


def _activation_summaries(endpoints):
    with tf.name_scope('summaries'):
        for act in endpoints.values():
            _activation_summary(act)
