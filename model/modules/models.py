import numpy as np
from keras.optimizers import RMSprop, Adam
from keras.models import  Model
from keras.layers import Input, Embedding, Reshape, LSTM, Masking, concatenate
from keras.layers import dot, merge
from keras.layers.core import Dense, Dropout, Lambda
from keras.regularizers import l2, l1
from keras import backend as k

#custom loss
from .losses import contrastive_loss,euclidean_distance


###################### EXTRACT A WANTED DIMENSION ####################
def crop(dimension, start, end,name=None):
    # Crops (or slices) a Tensor on a given dimension from start to end
    # example : to crop tensor x[:, :, 5:10]
    # call slice(2, 5, 10) as you want to crop on the second dimension
    def func(x):
        if dimension == 0:
            return x[start: end]
        if dimension == 1:
            return x[:, start: end]
        if dimension == 2:
            return x[:, :, start: end]
        if dimension == 3:
            return x[:, :, :, start: end]
        if dimension == 4:
            return x[:, :, :, :, start: end]

    return Lambda(func,name=name)
######################################################################


######################################################################
# Restructure the input to accept the data according to the model
def time_prediction_data_formatter(Data, Targets, train, test):
    return [Data[0][train],Data[1][train]],Targets[train],\
           [Data[0][test], Data[1][test]], Targets[test]
#####################################################################################



################################## PAPER VERSION ###################################################

def slice_embeddings(mode=1,name=None):
    def func(x):
#        assert (len(k.shape(x)) == 3)
        end = k.shape(x)[1]
        result = x
        if mode==1:
            result = x[:, 0:end - 1, :]
        elif mode == 2:
            result = x[:, end-1, :]

        return result

    return Lambda(func, name=name)
######################################################################
def build_path_model(settings):

    max_cascade_len = np.int32(settings['max_cascade_len'])
    users_range = np.int32(settings['users_range'])
    cascade_features = settings['cascade_features']
    n_factors = np.int32(settings['n_factors'])
    n_cells = np.int32(settings['n_cells'])
    dropout = np.float32(settings['dropout'])


    weights_loss = settings['weights_loss']
    weight_horizon_loss = np.float64(weights_loss[0])
    weight_emb_sim_loss = np.float64(weights_loss[1])

    n_features = 3
    #
    cascades_in = Input(shape=(max_cascade_len, n_features), name='cascade_in')
    user_in = crop(2, 0, 1)(cascades_in)
    time = crop(2, 1, 2)(cascades_in)
    delta_time = crop(2, 2, 3)(cascades_in)
    u = Embedding(input_dim=users_range, output_dim=n_factors, input_length=max_cascade_len,
                  embeddings_regularizer=l2(1e-4))(user_in)
    u = Reshape((max_cascade_len, n_factors))(u)
    # merge the features
    merged_layer = concatenate([u, time, delta_time], axis=2)

    lstm_layer = (
    LSTM(n_cells, input_shape=(None, max_cascade_len, n_features), return_sequences=True, stateful=False)(
        merged_layer))
    lstm_layer2 = (LSTM(n_cells, return_sequences=False, stateful=False)(lstm_layer))

    output = Dense(1, activation='sigmoid',name='activation_out')(lstm_layer2)


    users = slice_embeddings(mode=1,name='context_embeddings')(u)
    user = slice_embeddings(mode=2, name='target_user_embedding')(u)

    context = Lambda(lambda x: k.mean(x, axis=1), name="context_embedding")(users)

    dotprod = Lambda(lambda x: euclidean_distance(x), name="")([context, user])

    discriminant_output = Lambda(lambda x: k.exp(-x), name="embedding_similarity_out")(dotprod)

    model=Model(cascades_in, [output,discriminant_output])
    model.compile(optimizer='adam',
                  metrics={'activation_out': 'accuracy','embedding_similarity_out': 'accuracy'},
                  loss = {'activation_out': 'binary_crossentropy', 'embedding_similarity_out': 'binary_crossentropy'},
                    loss_weights = {'activation_out': weight_horizon_loss, 'embedding_similarity_out': weight_emb_sim_loss}
                  )
    return model
######################################################################
# Restructure the input to accept the data according to the model
def time_prediction_data_formatter(Data, Targets, train, test):
    return  Data[train],Targets[train],\
           Data[test], Targets[test]
#####################################################################################
