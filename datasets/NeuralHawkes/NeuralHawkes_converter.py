import pandas as pd
import numpy as np
import pickle

##
## Convert datasets to be fed into NeuralHawkes available at https://github.com/HMEIatJHU/neurawkes
##

def convert(df):
    data = []
    groups = df.groupby('item')

    cycle = 0.0
    cycles = len(groups)
    old_percent = 0.0

    print('Total cycles:', cycles)
    print('Conversion:   0.0%')

    for name, group in groups:
        percent = cycle / cycles

        if percent >= old_percent + 0.1:
            print("Conversion:  {0:.1f}%".format(percent * 100))
            old_percent = percent

        cycle += 1

        group.sort_values('timestamp')
        cascade = []
        last_time = 0
        first_iteration = True

        for row in group.itertuples():
            time = row.timestamp

            if first_iteration:
                first_iteration = False
            else:
                d = {'time_since_start': time, 'time_since_last_event': time - last_time, 'type_event': row.user}
                cascade.append(d)

            last_time = time

        data.append(cascade)

    print('Conversion: 100.0%')

    return data


if __name__ == '__main__':
    train = ' for instance twitter-large_train.dat'
    test = 'for instance twitter-large_test.dat'

    out_train = 'for instance train.pkl'
    out_dev = 'for instance dev.pkl'
    out_test = 'for instance test.pkl'

    dim_process=1719 #number of users for Flixster
    #dim_process=32043 #number of users for Twitter Large


    df_train = pd.read_csv(train, sep='\t')
    df_test = pd.read_csv(test, sep='\t')

    np.random.seed(101)
    msk = np.random.rand(len(df_train)) < 0.8
    df_dev = df_train[~msk]
    df_train = df_train[msk]
    df_dev.reset_index(drop=True)
    df_train.reset_index(drop=True)

    print('Converting train...')
    train_list = convert(df_train)
    print('Converting dev...')
    dev_list = convert(df_dev)
    print('Converting test...')
    test_list = convert(df_train)

    train_dict = {'test1': [], 'args': None, 'dim_process': dim_process, 'dev': [], 'train': train_list, 'test': []}
    dev_dict = {'test1': [], 'args': None, 'dim_process': dim_process, 'dev': dev_list, 'train': [], 'test': []}
    test_dict = {'test1': [], 'args': None, 'dim_process': dim_process, 'dev': [], 'train': [], 'test': test_list}

    with open(out_train, 'wb') as handle:
        pickle.dump(train_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)

    with open(out_dev, 'wb') as handle:
        pickle.dump(dev_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)

    with open(out_test, 'wb') as handle:
        pickle.dump(test_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)