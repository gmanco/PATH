from time import time
from keras.callbacks import Callback

class TimingCallback(Callback):
  def __init__(self):
    self.logs=[]
  def on_epoch_begin(self,epoch, logs={}):
    self.starttime=time()
  def on_epoch_end(self,epoch, logs={}):
    self.logs.append(time()-self.starttime)

