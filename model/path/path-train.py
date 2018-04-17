import json
import os,sys,inspect,getopt
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0,parentdir)
from time import time
########

from keras.callbacks import EarlyStopping,ModelCheckpoint,ReduceLROnPlateau, Callback
###
import model.modules.models as models
import model.modules.data_processor as data_processer
import parameters

def load_best_configuration_json(name):
    with open(name + '.json', 'rb') as f:
        return json.load(f)

class TimingCallback(Callback):
  def __init__(self):
    self.logs=[]
  def on_epoch_begin(self,epoch, logs={}):
    self.starttime=time()
  def on_epoch_end(self,epoch, logs={}):
    self.logs.append(time()-self.starttime)



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
        'predict-activation.py -i <dataset_name>'
        sys.exit(2)
    for opt, arg in opts:
        if opt == '-h':
            print
            'predict-activation-train.py -i <dataset_name>'
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

            ##load the best configuration found via Grid Search
            best_conf = load_best_configuration_json(data_path + 'configuration_' + dataset_name)

            print("Loaded the best configuration from grid search: ", best_conf)
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

            ##Preparing training data
            dp = data_processer.DataProcessor(settings)
            input_train, y_train = dp.build_data(tag_batch='train', tag_model='path',
                                                     min_size=min_size, max_size=max_size,
                                                     sample_size=sample_size)

            print("Building model...")
            model = models.build_path_model(model_settings)
            print(model.summary())
            cb = TimingCallback()
            callbacks_list = [
                EarlyStopping(
                    monitor='val_loss',
                    patience=parameters.earl_stop_patience,
                ),
                ModelCheckpoint(
                    filepath=models_path + 'model_{epoch:02d}-{val_loss:.4f}_' + train_dataset + '.h5',
                    monitor='val_loss',
                    save_best_only=True,
                )
                ,
                ReduceLROnPlateau(
                    monitor='val_loss',
                    factor=parameters.reduce_on_plateau_factor,
                    patience=parameters.reduce_on_plateau_patience,
                    min_lr=parameters.reduce_on_plateau_min_lr
                ), cb
            ]


            print("Fitting the network...\n")
            history = model.fit(input_train, [y_train,y_train],
                                epochs=nb_epochs,
                                batch_size=batch_size,
                                validation_split=validation_split, callbacks=callbacks_list)

            print("Fitting done.\n")
            print(cb.logs)


if __name__ == "__main__":
    main(sys.argv[1:])

