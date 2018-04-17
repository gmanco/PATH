from keras import backend as k



def euclidean_distance(inputs):
    assert len(inputs) == 2, \
        'Euclidean distance needs 2 inputs, %d given' % len(inputs)
    u, v = inputs
    return k.sqrt(k.sum((k.square(u - v)), axis=1, keepdims=True))

def contrastive_loss(y_true, y_pred):
    margin = 1.
    return k.mean(y_true * k.square(y_pred) + (1. - y_true) * k.square(k.maximum(margin - y_pred, 0.)))