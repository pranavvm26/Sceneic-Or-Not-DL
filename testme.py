"""
Test module to load TF Record file and that veryify data is being
serialized properly
"""

import os
import glob
import tensorflow as tf
import matplotlib.pyplot as plt


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
            'image_links': tf.FixedLenFeature([], tf.string),
        })

    image = tf.image.decode_jpeg(features['image_raw'], channels=3)

    # image_resize = tf.image.resize_images(image, (299, 299))
    label = tf.string_to_number(features['rating_average'], tf.float32)
    # label = features['rating_average']

    image_name = features['image_name']
    links = features['image_links']

    return image, label, image_name, links


def read_single():
    """
    Read first record of single file.
    Not unit test.
    """


    filename = glob.glob("rawdata/tfrecords_cleaned/*.tfrecords")


    filename_queue = tf.train.string_input_producer(filename, num_epochs=None)
    img, label, rk, links = read_and_decode_file(filename_queue)

    init = tf.global_variables_initializer()

    sess = tf.Session()
    sess.run(init)
    tf.train.start_queue_runners(sess=sess)

    for i in range(len(filename)):
        for k in range(50):
            img_1, label_1, rk_1, links_1 = sess.run([img, label, rk, links])

            print("ID: {0}, Lbl: {1}".format(k, label_1))

            plt.imsave("testimg_{0}.jpg".format(k), img_1)
            print("write", k)


if __name__ == '__main__':
    read_single()
