"""
    Cleans up raw dataset and downloads image links to a local db
"""
import pandas as pd
import requests
from tqdm import tqdm
import multiprocessing as mp
from bs4 import BeautifulSoup
from sqlalchemy.orm import sessionmaker
from dataextraction.model import Base, engine, SoNData


def parse_link_for_image(source_url):
    """

        :param source_url:
        :return:
    """

    response = requests.get(source_url)

    soup = BeautifulSoup(response.text, 'html.parser')
    img_tags = soup.find_all('img')

    source_name_img = []
    for img in img_tags:
        if 's0' in img['src']:
            source_name_img.append((img['alt'], img['src']))

    assert len(source_name_img) == 1, "more than a single source image found in: {0}".format(source_url)

    return source_name_img


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
                        df_raw['Geograph URI'].tolist()))

    # chuck the data into n threads
    k, m = divmod(len(data_rawlist), chuck_size)
    return list(data_rawlist[i * k + min(i, m):(i + 1) * k + min(i + 1, m)] for i in range(chuck_size))


def mp_process_links(tid, id_url_tuple, mp_list):
    """

        :param id_link_tuple:
        :param mp_list:
        :return:
    """

    for idx, url in tqdm(id_url_tuple, desc=tid):
        try:
            name_and_imagelink = parse_link_for_image(source_url=url)
            mp_list.append((idx, url, name_and_imagelink[0][0], name_and_imagelink[0][1]))
        except Exception as e:
            print("exception with thread {0}, for row: {1} and url: {2}".format(tid, idx, url))
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
    df_rawdata = df_rawdata.iloc[:100000]

    threads = 5

    # prep data for multiprocessing
    data_thread_chunks = prep_for_multiprocessing(df_rawdata, threads)

    # create a shared multiprocessing dict
    mp_list = mp.Manager().list()

    # multiprocessing list - process instantiation
    processes = []
    for i in range(threads):
        processes.append(mp.Process(name="Thread {0}".format(i),
                                    target=mp_process_links,
                                    args=("Thread {0}".format(i),
                                          data_thread_chunks[i],
                                          mp_list,)))

    print("\n starting processess ...\n")
    # process start
    for process in processes:
        process.start()

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