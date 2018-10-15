# -*- coding: utf-8 -*-
"""
Created on Sat Sep 29 09:28:37 2018

@author: Mauro_Sergio
"""
#import numpy as np
import pandas as pd
from keras.models import model_from_json

entrada = pd.read_csv('inputXOR.csv')

arquivo = open('classificador_xor.json', 'r')
estrutura_rede = arquivo.read()
arquivo.close()

model = model_from_json(estrutura_rede)
model.load_weights('classificador_xor.h5')

novo = entrada[600:]

previsao = model.predict(novo)
previsao = (previsao > 0.5)

previsores = pd.read_csv('entradas_breast.csv')
classe = pd.read_csv('saidas_breast.csv')
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['binary_accuracy'])
resultado = model.evaluate(previsores, classe)