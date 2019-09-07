import csv
import tensorflow as tf
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import array_ops


def read_and_decode_file(filename_queue):
    """
    Read and decode a tensorflow records file
    :param filename_queue:  Filename to load
    :param jitter: Boolean to implement random scale jittering of images
    :param augment_data:: Boolean to augment data set
    :param augment_classes: Which classes to augment
    :param augment_factor: Numbner of times to create copies of augmented images
    :return: image(numpy uint8), label (int), rowkey(string)
    """

    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)
    features = tf.parse_single_example(
        serialized_example, features={
            'image_raw': tf.FixedLenFeature([], tf.string),
            'rating_average': tf.FixedLenFeature([], tf.string),
            'image_name': tf.FixedLenFeature([], tf.string),
        })

    image = tf.image.decode_jpeg(features['image_raw'], channels=3)
    image_resize = tf.image.resize_images(image, (299, 299))
    image_resize = tf.divide(image_resize, 255)
    label = tf.round(tf.string_to_number(features['rating_average'], tf.float32))
    image_name = features['image_name']

    return image_resize, label, image_name


def input_pipeline(filenames, threads, dequeue_leftover, batchsize, epochs=1):

    with tf.device('/cpu:0'):

        filename_queue = tf.train.string_input_producer(filenames, shuffle=True, num_epochs=epochs)

        image, label, image_name = read_and_decode_file(filename_queue)

        capacity = dequeue_leftover + (3 + threads) * batchsize

        images_batch, labels_batch, imgnames_batch = tf.train.shuffle_batch(
            [image, label, image_name], batch_size=batchsize,
            capacity=capacity,
            min_after_dequeue=dequeue_leftover,
            num_threads=threads)

        return images_batch, labels_batch, imgnames_batch
