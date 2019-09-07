import tensorflow as tf
import numpy as np
from tqdm import tqdm
import glob
import os
from datetime import datetime
from dataextraction.recordwriter import write_tfrecord

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
            'rating_variance': tf.FixedLenFeature([], tf.string),
            'latitude': tf.FixedLenFeature([], tf.string),
            'longitude': tf.FixedLenFeature([], tf.string),
            'geography_url': tf.FixedLenFeature([], tf.string),
            'image_links': tf.FixedLenFeature([], tf.string),
        })

    image = tf.image.decode_jpeg(features['image_raw'], channels=3)
    rating_avg = features['rating_average']
    image_name = features['image_name']
    rating_variance = features['rating_variance']
    latitude = features['latitude']
    longitude = features['longitude']
    geography_url = features['geography_url']
    image_links = features['image_links']

    return image, rating_avg, image_name, rating_variance, latitude, longitude, geography_url, image_links

def clean_remake_tfrecords(filenames, record_dir, newfilename):

    stack_size = 45
    h, w = 500, 500

    # initialize empty numpy stack arrays
    image_stack = np.empty([stack_size, h, w, 3])
    average_stack = np.empty([stack_size], dtype='S100')
    imgname_stack = np.empty([stack_size], dtype='S250')
    variance_stack = np.empty([stack_size], dtype='S100')
    latitude_stack = np.empty([stack_size], dtype='S100')
    longitude_stack = np.empty([stack_size], dtype='S100')
    geourl_stack = np.empty([stack_size], dtype='S500')
    imglinks_stack = np.empty([stack_size], dtype='S500')

    # counter
    cnt = 0

    # file CNT
    filecnt = 0

    filename_queues = tf.train.string_input_producer(filenames, num_epochs=None)
    image, rating_avg, image_name, rating_variance, latitude, longitude, geography_url, image_links = read_and_decode_file(filename_queues)

    init = tf.global_variables_initializer()

    sess = tf.Session()
    sess.run(init)
    tf.train.start_queue_runners(sess=sess)

    for i in tqdm(range(len(filenames))):
        for k in range(50):
            img, rat_avg, img_name, rat_var, lat, long, geo_url, img_link = sess.run([image, rating_avg, image_name,
                                                                                       rating_variance, latitude,
                                                                                       longitude, geography_url,
                                                                                       image_links])

            try:
                assert img.shape == (500, 500, 3), "image shape does not comply: {0}".format(img.shape)

                # add to stack
                image_stack[cnt] = img
                average_stack[cnt] = float(rat_avg)
                imgname_stack[cnt] = img_name
                variance_stack[cnt] = float(rat_var)
                latitude_stack[cnt] = float(lat)
                longitude_stack[cnt] = float(long)
                geourl_stack[cnt] = geo_url
                imglinks_stack[cnt] = img_link

                # increment count
                cnt += 1
            except Exception as e:
                # print("Exception =====> {0}".format(e))
                pass

            if cnt == stack_size:
                write_tfrecord(images=image_stack,
                               avg_ratings=average_stack,
                               image_names=imgname_stack,
                               rating_variance=variance_stack,
                               latitude=latitude_stack,
                               longitude=longitude_stack,
                               geography_url=geourl_stack,
                               image_links=imglinks_stack,
                               record_dir=record_dir,
                               filename=newfilename + '{0}-{1}'.format(filecnt,
                                                                       datetime.now().strftime("%y%m%d%H%M%S%f")))

                # initialize empty numpy stack arrays
                image_stack = np.empty([stack_size, h, w, 3])
                average_stack = np.empty([stack_size], dtype='S100')
                imgname_stack = np.empty([stack_size], dtype='S250')
                variance_stack = np.empty([stack_size], dtype='S100')
                latitude_stack = np.empty([stack_size], dtype='S100')
                longitude_stack = np.empty([stack_size], dtype='S100')
                geourl_stack = np.empty([stack_size], dtype='S500')
                imglinks_stack = np.empty([stack_size], dtype='S500')

                # counter
                cnt = 0

                filecnt += 1


if __name__ == "__main__":
    filenames_ = glob.glob("rawdata/tfrecords/*.tfrecords")
    record_dir_ = "rawdata/tfrecords_cleaned"
    os.makedirs(record_dir_, exist_ok=True)
    new_filename_ = "dataset-clean-scenic-or-not-encoded"
    clean_remake_tfrecords(filenames=filenames_, record_dir=record_dir_, newfilename=new_filename_)