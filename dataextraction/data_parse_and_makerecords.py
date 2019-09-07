"""
    Cleans up raw dataset and downloads image links to a local db
"""
import os
import time
from io import BytesIO
from PIL import Image
import numpy as np
import requests
import warnings
import pandas as pd
import cv2
import matplotlib.pyplot as plt
import multiprocessing as mp
from bs4 import BeautifulSoup
from datetime import datetime
from sqlalchemy.orm import sessionmaker
from dataextraction.model import Base, engine, SoNData
from dataextraction.recordwriter import write_tfrecord

warnings.filterwarnings("ignore")


def parse_link_for_image(source_url):
    """

        :param source_url:
        :return:
    """

    response = requests.get(source_url, verify=False)

    soup = BeautifulSoup(response.text, 'html.parser')
    img_tags = soup.find_all('img')

    source_name_img = []
    for img in img_tags:
        if 's0' in img['src']:
            source_name_img.append((img['alt'], img['src']))

    assert len(source_name_img) == 1, "more than a single source image found in: {0}".format(source_url)

    response = requests.get(source_name_img[0][1], verify=False)
    img_scene = np.asarray(Image.open(BytesIO(response.content)))

    return img_scene, source_name_img[0][1], source_name_img[0][0]


def push_to_database(df_raw):
    """

        :param df_raw:
        :return:
    """
    # Create all tables in the engine. This is equivalent to
    # "Create Table if not exist" statements in raw SQL.
    Base.metadata.create_all(engine)

    # create a new db session
    dbsess = sessionmaker(bind=engine)
    session = dbsess()

    # Insert a Person in the person table
    for i, data in df_raw.iterrows():
        row = SoNData(img_latitude=data['Lat'],
                      img_longitude=data['Lon'],
                      img_rating_average=data['Average'],
                      img_rating_variance=data['Variance'],
                      img_voting_count=data['Votes'],
                      img_source_link=data['Geograph URI'],
                      img_link=data['img_link'],
                      img_name=data['img_name'], )
        # equivalent to insert
        session.add(row)
    # commit all inserts
    session.commit()

    return 0


def prep_for_multiprocessing(df_raw, chuck_size):
    """

        :param df_raw:
        :param chuck_size:
        :return:
    """

    data_rawlist = list(zip(df_raw['ID'].tolist(),
                            df_raw['Geograph URI'].tolist(),
                            df_raw['Lat'].tolist(),
                            df_raw['Lon'].tolist(),
                            df_raw['Average'].tolist(),
                            df_raw['Variance'].tolist(),
                            df_raw['Votes'].tolist()))

    # chuck the data into n threads
    k, m = divmod(len(data_rawlist), chuck_size)
    return list(data_rawlist[i * k + min(i, m):(i + 1) * k + min(i + 1, m)] for i in range(chuck_size))


def mp_process_links(tid, data_tuples, mp_list, filename, record_dir):
    """

    :param tid:
    :param data_tuples:
    :param mp_list:
    :param filename:
    :param record_dir:
    :return:
    """
    stack_size = 50
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

    for idx, url, lat, lon, avg, var, votes in data_tuples:
        try:
            img, link, name = parse_link_for_image(source_url=url)

            img__ = cv2.resize(img, (h, w))

            assert img__.shape == (h, w, 3), "mishaped: {0}".format(img__.shape)

            # add to stack
            image_stack[cnt] = img__
            average_stack[cnt] = avg
            imgname_stack[cnt] = name
            variance_stack[cnt] = var
            latitude_stack[cnt] = lat
            longitude_stack[cnt] = lon
            geourl_stack[cnt] = url
            imglinks_stack[cnt] = link

            # increment count
            cnt += 1

            print("{0} {1}".format(tid, cnt))

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
                               filename=filename+'-{0}-{1}'.format(tid,
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

            mp_list.append((idx, url, link, name))

        except Exception as e:
            print("exception {0} with thread {1}, for row: {2} and url: {3}".format(e, tid, idx, link))
            mp_list.append((idx, url, None, None))
            pass

    return mp_list


def merge_data(df_raw, processed_list):
    """

        :param df_raw:
        :param processed_list:
        :return:
    """
    df_processed = pd.DataFrame(list(processed_list), columns=['ID', 'Geograph URI', 'img_name', 'img_link'])
    df_raw_merged = pd.merge(df_raw, df_processed, on=['ID', 'Geograph URI'], how='left')
    assert len(df_raw_merged.index) == len(df_raw.index), "unequal lengths post merge"
    return df_raw_merged


def main():
    """

        :return:
    """

    # read source tsv
    df_rawdata = pd.read_table("rawdata/votes.tsv")
    df_rawdata = df_rawdata.iloc[100000:]

    # process meta
    threads = 10
    filename = "dataset-scenic-or-not-encoded"
    record_dir = "rawdata/tfrecords"
    os.makedirs(record_dir, exist_ok=True)

    # prep data for multiprocessing
    data_thread_chunks = prep_for_multiprocessing(df_rawdata, threads)

    # create a shared multiprocessing dict
    mp_list = mp.Manager().list()

    # multiprocessing list - process instantiation
    processes = []
    for i in range(threads):
        processes.append(mp.Process(name="thread-{0}".format(i),
                                    target=mp_process_links,
                                    args=("thread-{0}".format(i),
                                          data_thread_chunks[i],
                                          mp_list, filename, record_dir,)))

    print("\n starting processess ...\n")
    # process start
    for ipx, process in enumerate(processes):
        print("starting thread {0}".format(ipx))
        process.start()
        time.sleep(2)

    # process join
    for process in processes:
        # wait for processes to complete
        process.join()

        # convert into a pure python list
        processed_list = [x for x in mp_list]

        # sort it
        processed_list = sorted(processed_list, key=lambda x: x[0])

    df_raw_processed = merge_data(df_raw=df_rawdata,
                                  processed_list=processed_list)

    # push to a sqllite db
    print("\n pushing data to a local sqlite db ...\n")
    push_to_database(df_raw=df_raw_processed)


if __name__ == "__main__":
    main()
    print('Done!')