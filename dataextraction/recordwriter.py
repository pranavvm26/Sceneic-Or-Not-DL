import os
import cv2
import numpy as np
import tensorflow as tf
from datetime import datetime


def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def _float_feature(value):
    return tf.train.Feature(int64_list=tf.train.FloatList(value=[value]))


def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def write_tfrecord(images, avg_ratings, image_names, rating_variance, latitude, longitude, geography_url, image_links,
                   record_dir, filename):

    num_examples = images.shape[0]

    assert num_examples == images.shape[0] == avg_ratings.shape[0] == len(image_names) == \
           rating_variance.shape[0] == latitude.shape[0] == longitude.shape[0] == len(geography_url) \
           == len(image_links), "Shapes of params dont match"

    os.makedirs(record_dir, exist_ok=True)

    filename = os.path.join(record_dir, filename + '.tfrecords')

    print("\n [{0}] writing {1} with reference length {1}".format(datetime.now().strftime("%Y-%m-%d %I:%M:%S %p"),
                                                                  filename,
                                                                  num_examples))

    writer = tf.python_io.TFRecordWriter(filename)

    for index in range(num_examples):

        img = cv2.cvtColor(images[index].astype(np.uint8), cv2.COLOR_BGR2RGB)
        _, img_encoded = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 99])
        img_encoded = img_encoded.tostring()

        example = tf.train.Example(features=tf.train.Features(feature={
            'image_raw': _bytes_feature(img_encoded),
            'rating_average': _bytes_feature(avg_ratings[index]),
            'image_name': _bytes_feature(image_names[index]),
            'rating_variance': _bytes_feature(rating_variance[index]),
            'latitude': _bytes_feature(latitude[index]),
            'longitude': _bytes_feature(longitude[index]),
            'geography_url': _bytes_feature(geography_url[index]),
            'image_links': _bytes_feature(image_links[index]),
        }))
        writer.write(example.SerializeToString())

    writer.close()