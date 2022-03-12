from sklearn.preprocessing import MinMaxScaler

import pandas as pd
import struct, socket
from time import time
from os import listdir
from sys import argv
from random import randint


COLUMNS = ['pkSeqID','stime','flgs','proto','saddr','sport','daddr','dport',
           'pkts','bytes','state','ltime','seq','dur','mean','stddev','smac',
           'dmac','sum','min','max','soui','doui','sco','dco','spkts','dpkts',
           'sbytes','dbytes','rate','srate','drate','attack','category','subcategory' ]

DTYPES = {
    "proto":      "string",
    "saddr":      "string",
    "sport":      "string",
    "daddr":      "string",
    "dport":      "string",
    "pkts":        "int64",
    "state":      "string",
    "dur":       "float64",
    "spkts":       "int64",
    "dpkts":       "int64",
    "sbytes":      "int64",
    "dbytes":      "int64",
    "srate":     "float64",
    "drate":     "float64",
    "attack":      "int64"
}

SRC_DATA_PATH = '../Data/Entries/'
PREPARED_DATA_PATH = '../Data/balanced_data.csv'

def preprocess_dataset(df):
    def ip2long(ip):
        try:
            packedIP = socket.inet_aton(ip)
        except OSError:
            packedIP = socket.inet_aton('192.168.0.1')
        return struct.unpack("!L", packedIP)[0]

    def str2int(s):#перевод порта в int
        if type(s) == pd._libs.missing.NAType:
            return 0
        if s[:2] == '0x':
            return int(s, base = 16)
        else:
            return int(s)
    
    df['sport'] = df['sport'].apply(str2int)
    df['dport'] = df['dport'].apply(str2int)

    df['saddr'] = df['saddr'].apply(ip2long)
    df['daddr'] = df['daddr'].apply(ip2long)

    return df


if __name__ == "__main__":
    clean_df = pd.DataFrame()
    dirty_df = pd.DataFrame()
    
    time_start = time()

    list_dir = sorted(listdir(SRC_DATA_PATH))

    if len(argv) <= 1:
        index_of_dirty_dataset = randint(0, len(list_dir) - 1)
    else:
        index_of_dirty_dataset = randint(0, int(argv[1]) - 1)

    print('Dirty dataset will be {}'.format(index_of_dirty_dataset))

    # Concat all datasets to one big, but only with 'attack == 0'
    for i, dataset_name in enumerate(list_dir):
        dataset_path = SRC_DATA_PATH + dataset_name
        
        print('Reading \'{}\' as clean'.format(dataset_path), end=' ')
        df = pd.read_csv(dataset_path, 
                     names = COLUMNS, 
                     dtype = DTYPES)
    
        clean_df = pd.concat([clean_df, df[df.attack == 0]])

        print('(+{})'.format(len(df[df.attack == 0])))

        if i == index_of_dirty_dataset:
            print('Reading \'{}\' as dirty'.format(dataset_path))

            dirty_df = df[df.attack == 1]

        if (len(argv) > 1) and (i > int(argv[1])):
            break
    
    print('Total time reading: {}'.format(time() - time_start))

    if len(dirty_df) > 2 * len(clean_df):
        print('Cut dirty dataset: {} -> {}'.format(len(dirty_df), 2 * len(clean_df)))
        dirty_df = dirty_df[:2 * len(clean_df)]

    all_df = pd.concat([clean_df, dirty_df])
    all_df = preprocess_dataset(all_df)

    clean_df = all_df[all_df.attack == 0]
    dirty_df = all_df[all_df.attack == 1]

    print('Clean sampels: {}\nDirty samples: {}'.format(len(clean_df), len(dirty_df)))

    print('Writting \'{}\''.format(PREPARED_DATA_PATH))
    all_df.to_csv(PREPARED_DATA_PATH, index=False)

    if len(all_df) != len(all_df.drop_duplicates()):
        print('[WARNING] There are {} duplicate lines' \
            .format(len(all_df) - len(all_df.drop_duplicates())))
