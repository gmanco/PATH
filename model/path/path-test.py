import json
import os,sys,inspect, getopt
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0,parentdir)
########

from keras.models import load_model

import model.modules.data_processor as data_processor
import parameters
import pandas as pd

import numpy as np
from keras import backend as k
from model.modules.losses import euclidean_distance

def load_best_configuration_json(name):
    with open(name + '.json', 'rb') as f:
        return json.load(f)
##


def main(argv):
    global dataset_name
    global dataset
    global train_dataset
    global test_dataset
    global dev_dataset
    global data_path
    global models_path
    global results_path
    global train_file
    global test_file
    global dev_file
    global model_figure
    try:
        opts, args = getopt.getopt(argv, "hi:o:", ["ifile="])
    except getopt.GetoptError:
        print
        'path-test.py -i <dataset_name>'
        sys.exit(2)
    for opt, arg in opts:
        if opt == '-h':
            print
            'path-test.py -i <dataset_name>'
            sys.exit()
        elif opt in ("-i", "--ifile"):
            dataset_name = arg
            print("Dataset Name is ", dataset_name)
            ####################
            dataset = dataset_name + '/'
            train_dataset = dataset_name + "_train"
            test_dataset = dataset_name + "_test"
            dev_dataset = dataset_name + "_dev"
            ####################
            data_path = '../../datasets/' + dataset
            models_path = './models/' + dataset
            results_path = './results/' + dataset
            train_file = "train/" + train_dataset
            test_file = "test/" + test_dataset
            dev_file = "dev/" + dev_dataset
            model_figure = models_path + 'model.png'
            print("Train File=", train_file)
            print("Test File=", test_file)

            ##output will be saved in these files
            loss_file_name = results_path + parameters.model_for_prediction + '_path_loss.csv'

            ##load the best configuration
            best_conf = load_best_configuration_json(data_path + 'configuration_' + dataset_name)


            print("Loaded the best configuration: ", best_conf)
            model_settings = best_conf['model']
            training_settings = best_conf['fitting']
            data_settings = best_conf['data']

            ##Train settings
            if 'epochs' in training_settings:
                nb_epochs = training_settings['epochs']
            else:
                nb_epochs = 5
            if 'batch_size' in training_settings:
                batch_size = training_settings['batch_size']
            else:
                batch_size = 128
            if 'validation_split' in training_settings:
                validation_split = training_settings['validation_split']
            else:
                validation_split = 0.3

            ##Data Settings
            min_size = data_settings['min_size']
            max_size = data_settings['max_size']
            sample_size = data_settings['sample_size']

            # Data processor settings
            settings = {
                'path_rawdata': data_path,
                'ratio_train': '1',
                'to_read': {'train': train_file, 'test': test_file},
                'partial_predict': 'False',
                'look_ahead': parameters.look_ahead
            }

            dp = data_processor.DataProcessor(settings)

            model = load_model(models_path + parameters.model_for_prediction, custom_objects={'k': k, 'euclidean_distance':euclidean_distance})
            print("Model loaded from file=", parameters.model_for_prediction)
            print(model.summary())
            ## Test data
            test_x, test_y = dp.build_data(tag_batch='test', tag_model='path', min_size=min_size,
                                           max_size=max_size, sample_size=sample_size)
            print("Starting prediction.....")
            # Collect the prediction
            predicted_prob = model.predict(test_x)
            predictions_1 = np.concatenate((test_y, predicted_prob[0]),
                                           axis=1)
            df_pred = pd.DataFrame(predictions_1)
            df_pred.columns = ['active', 'prob_active']
            df_pred.to_csv(loss_file_name, sep='\t', index=False)
            print("Prediction done!")


if __name__ == "__main__":
    main(sys.argv[1:])









