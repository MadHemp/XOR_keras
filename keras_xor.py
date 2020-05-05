# -*- coding: utf-8 -*-
"""
Created on Sat Sep 29 09:28:37 2018

@author: Mauro_Sergio
"""

import pandas as pd
from keras.models import Sequential
from keras.layers import Dense

x = pd.read_csv('./data/input.csv')
y = pd.read_csv('./data/output.csv')

'''
Criando modelo de Rede Neural
'''
model = Sequential()
model.add(Dense(10, input_dim = 2, activation = 'relu'))
model.add(Dense(6, activation = 'relu'))
model.add(Dense(1,activation = 'sigmoid'))

model.compile(loss = 'mean_squared_error', optimizer = 'adam', metrics = ['accuracy'])
model.fit(x , y, epochs=100, batch_size = 32)

'''
Realizando predições
'''
predicted = model.predict(x)
predicted = predicted > 0.5
print(predicted)

model.save('classificador_xor.h5')
