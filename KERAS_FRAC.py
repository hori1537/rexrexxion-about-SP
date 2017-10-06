'''Trains a simple convnet on the MNIST dataset.

Gets to 99.25% test accuracy after 12 epochs
(there is still a lot of margin for parameter tuning).
16 seconds per epoch on a GRID K520 GPU.
'''

from __future__ import print_function
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras import backend as K
from keras.layers import Activation
from sklearn.model_selection import train_test_split
import numpy as np
import numpy

import time
import pandas as pd
import keras
from keras.utils import plot_model
import pandas
import pandas as pd


#Date from INTERLAD

from Setting_Param import *
#Local_Setting_Param

raw_input = pandas.read_csv(open(str(ADDRESS)+str(CSV_NAME_FRAC)))
#raw_input = numpy.loadtxt(open(str(ADDRESS)+str(CSV_NAME_FRAC)), delimiter=",",skiprows=1)

[information_p, component_p, parameter_p]  = np.hsplit(raw_input, [20,82])

component = np.array(component_p)
parameter = np.array(parameter_p)


#[component_a,component]= numpy.vsplit(component_all,[0])
#[parameter_a,parameter]= numpy.vsplit(parameter_all,[0])


#print("parameter_a",parameter_a)
#print("parameter",parameter)
#print("component_a",component_a)
#print("component",component)

TRAIN_DATA_SIZE=DATA_SIZE_FRAC-10
#DATE_SIZE_FRAC =2448

#parameter_train, parameter_test, component_train, component_test = train_test_split(parameter, component, test_size=0.05, random_state=0)

[parameter_train, parameter_test] = numpy.vsplit(parameter, [TRAIN_DATA_SIZE])
[component_train, component_test] = numpy.vsplit(component, [TRAIN_DATA_SIZE])

batch_size = 4000
epochs = 300000

es_cb = keras.callbacks.EarlyStopping(monitor='val_loss', patience=3000, verbose=1, mode='auto')



model = Sequential()
model.add(Dense(DATA_SIZE_FRAC, input_dim=62, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(300, activation = 'relu'))
model.add(Dropout(0.2))
model.add(Dense(300, activation = 'relu'))
model.add(Dropout(0.2))
model.add(Dense(1))

#model.add(Dense(300, activation = None))
#0.0587
##0.0929 NN has None layer got bad
#0.3937

model.compile(loss='mse',
              optimizer=keras.optimizers.Adam(),
              metrics=['accuracy'])

model.fit(component_train, parameter_train,
          batch_size=batch_size,
          epochs=epochs,
          verbose=1,
          validation_data=(component_test, parameter_test),
          callbacks=[es_cb])
          
score = model.evaluate(component_test, parameter_test, verbose=0)
parameter_predict=model.predict(component_test, batch_size=32, verbose=1)


model.save('C:\Deeplearning/model_FRAC_' +str(score[0]) + '.h5')
#plot_model(model, to_file='C:\Deeplearning/model.png')
#plot_model(model, to_file='model.png')


print('Test loss:', score[0])
print('Test accuracy:', score[1])

print ('predict', parameter_predict)

print('C:\Deeplearning/model_FRAC_' +str(score[0]) + '.h5')
