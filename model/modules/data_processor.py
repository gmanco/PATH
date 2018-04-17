import numpy
import random
import pandas as pd
import os
from keras.preprocessing import sequence as seq_padder


dtype = numpy.float32
utype = numpy.int32
'''
    This class process raw data and prepare them according to one of the PATH model:
     - Predict whether a specific user will be active before a horizon time T
    '''
class DataProcessor(object):

####################################################################################################################
    ## Receives the parameters from the calling method
    def __init__(self, settings):
        print("Initialize the data processor ... ")
        self.path_rawdata = os.path.abspath(settings['path_rawdata'])
        #
        self.to_read = settings['to_read']
        self.look_ahead=settings['look_ahead']
        self.ratio_train = numpy.float32(settings['ratio_train'])
        self.data = {
            'train': [],
            'dev': [],
            'test': []
        }
        for tag_split in self.to_read.keys():
            print("Reading data for tag : ", tag_split)
            path_to_read = self.path_rawdata + '/' + self.to_read[tag_split] + '.dat'
            ###
            ### READ THE CASCADES
            self.data[tag_split] = self._process_cascades_(pd.read_csv(path_to_read, sep="\t"))
        self.events = {}
        self.time_horizon = 0
        #
        for seq in self.data['train']:
            for item in seq:
                type_event = item['type_event']
                if type_event in self.events:
                    self.events[type_event] += 1
                else:
                    self.events[type_event] = 1
                if self.time_horizon < item['time_since_start']:
                    self.time_horizon = item['time_since_start']
        self.dim_process = numpy.int32(len(self.events) )
        self.event_range = numpy.int32(max(numpy.array(list(self.events.keys()))) + 1)

        # We assume that the time horizon (the time within we want to made the prediction)
        #  is (look_ahead %) further away than the last event. This parameter is read from Parameters.py
        self.time_horizon = self.look_ahead*self.time_horizon
        #
        len_to_select = numpy.int32(len(self.data['train']) * self.ratio_train)
        self.data['train'] = self.data['train'][:len_to_select]
        #
        self.lens = {
            'train': len(self.data['train']),
            'test': len(self.data['test']),
            'dev': len(self.data['dev'])

        }

        #
        # decide whether we predict a partial cascade
        self.partial_predict = settings['partial_predict']
        self.substream = [0]
        '''
        for now, partial predict is only to predict event-0
        in the future, it can be more general
        like predicting a substream 0, 2, 4, ...
        this feature is useless now ...
        '''
        #
        print("Finished data processer initialization")
####################################################################################################################

####################################################################################################################
    ## Process all cascades and group them: this is called in init and the result goes to self.data
    ##
    def _process_cascades_(self, cascades, normalize=True):
        cascades = cascades[['user', 'item', 'timestamp']].sort_values(by=['item', 'timestamp'], ascending=[True, True])
        if normalize:
            cols_to_norm = ['timestamp']
            cascades[cols_to_norm] = cascades[cols_to_norm].apply(lambda x: (x - x.min()) / (x.max() - x.min()))

        previtem = -1
        prevtimestamp = 0
        grouped_data = []
        seq = []
        for index, row in cascades.iterrows():
            curitem = row['item']
            user = row['user']
            time = row['timestamp']
            if curitem != previtem:
                event = {'type_event': user, 'time_since_start': time, 'time_since_last_event': 0}
                if seq != []:
                    grouped_data.append(seq)
                seq = [event]
                previtem = curitem
            else:
                event = {'type_event': user, 'time_since_start': time, 'time_since_last_event': time - prevtimestamp}
                seq.append(event)
            prevtimestamp = time
        if seq != []:
            grouped_data.append(seq)
            #
        print("# of cascades=", len(grouped_data))
            #

        return grouped_data
####################################################################################################################



################################################################################################
    ## Generic call to the data builder
    ## Tag model defines which method is used to construct the sequences
    ##
    def build_data(
        self, tag_batch, tag_model,
        min_size=1, max_size=100,
        sample_size=None
    ):
        #
        self.tag_batch = tag_batch
        self.tag_model=tag_model
        self.min_size=min_size
        self.max_size=max_size


        print("Constructing the sequences for model=",tag_model, " type=",tag_batch)

        if tag_model == 'path':
            return self.build_data_path(min_size, max_size, sample_size)
        else:
            print("Model not yet implemented")

        print("Finished building data matrices.")
####################################################################################################################


    def build_data_path(self, min_size, max_size, sample_size):
        samples = []
        targets = []

        self.max_len=max_size

        all_users = numpy.array(list(self.events.keys()))

        time_horizon = self.time_horizon
        num = 0
        for sequence in self.data[self.tag_batch]:
            if len(sequence)>=min_size and len(sequence)<=max_size:
                num=num+1
                active_users =numpy.array([user['type_event'] for user in sequence])
                user_timestamps = numpy.array([time['time_since_start'] for time in sequence])

                deltas_timestamps = [[(user_timestamps[idx]- user_timestamps[idx - 1])] for idx in
                                    range(1, len(sequence))]
                deltas_timestamps = [[0.0]] + deltas_timestamps
                deltas_timestamps = numpy.array(deltas_timestamps)
                ########################################################################################

                ##Set of non active users
                inactive_users = numpy.setdiff1d(all_users, active_users)

                begin_sequence = True
                bases_increasing_length = []
                for i, user_act in enumerate(active_users):
                    ## first user of a new sequence (hence a new base has to be created)
                    if begin_sequence:
                        base_one = numpy.array([[user_act, user_timestamps[i], deltas_timestamps[i]]])
                        bases_increasing_length.append(base_one)
                        begin_sequence = False
                    else:
                        bs = bases_increasing_length[-1]

                        new_sequence_false = numpy.vstack(
                            (bs, numpy.array([user_act, user_timestamps[i - 1], deltas_timestamps[i - 1]])))
                        samples.append(new_sequence_false)
                        targets.append([0.0])

                        ##############################################################################################################
                        new_sequence_true = numpy.vstack(
                            (bs, numpy.array([user_act, time_horizon, (time_horizon-deltas_timestamps[i])])))
                        samples.append(new_sequence_true)
                        targets.append([1.0])

                      ##############################################################################################################


                        if(self.tag_batch=='test'):
                            bs[i-1][1]=user_timestamps[i-1]
                            bs[i-1][2]=deltas_timestamps[i-1]

                            new_sequence_true = numpy.vstack(
                                (bs, numpy.array([user_act, user_timestamps[i], deltas_timestamps[i]])))
                            samples.append(new_sequence_true)
                            targets.append([1.0])
                            new_sequence_true = numpy.vstack((bs, numpy.array([user_act, time_horizon, (time_horizon-deltas_timestamps[i])])))
                            samples.append(new_sequence_true)
                            targets.append([1.0])
                        else:
                            new_sequence_true = numpy.vstack((bs, numpy.array([user_act, user_timestamps[i], deltas_timestamps[i]])))
                            samples.append(new_sequence_true)
                            targets.append([1.0])

                        bases_increasing_length.append(new_sequence_true)

                        if (i == len(active_users) - 1):
                            time_last_active = user_timestamps[i]
                            #
                            if len(inactive_users) <= sample_size:
                                sampled_inactive = list(inactive_users)
                            else:
                                sampled_inactive = random.sample(list(inactive_users), sample_size)
                            # NON ACTIVE USERS
                            for user_non_active in sampled_inactive:
                                # for user_non_active we use time_horizon (i.e., max timestamp in the current cascade + a look_ahead_horizon)
                                new_sequence_sampled_non_active_user = numpy.vstack(
                                    (bs, numpy.array([user_non_active, time_horizon, (time_horizon - time_last_active)])))
                                samples.append(new_sequence_sampled_non_active_user)
                                targets.append([0.0])
        temp_users = [[x[0] for x in y] for y in samples]
        temp_timestamps = [[x[1] for x in y] for y in samples]
        temp_deltas_timestamps = [[x[2] for x in y] for y in samples]
        padded_users = seq_padder.pad_sequences(temp_users, max_size)
        padded_timestamps = seq_padder.pad_sequences(temp_timestamps, max_size, dtype=float)
        padded_deltas_timestamps = seq_padder.pad_sequences(temp_deltas_timestamps, max_size, dtype=float)
        #########################################################################################################
        samples = numpy.array([numpy.array([numpy.array([a, b, c]) for a, b, c in zip(xxx, yyy, zzz)]) for xxx, yyy, zzz in
                            zip(padded_users, padded_timestamps, padded_deltas_timestamps)])
        samples = numpy.array(samples)
        targets = numpy.array(targets)
        #
        return samples, targets