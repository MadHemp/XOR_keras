# -*- coding: utf-8 -*-
"""
Created on Sat Sep 29 09:28:37 2018

@author: Mauro_Sergio
"""

'''
Carregando pesos da Rede Neural para fazer previsões
'''
import pandas as pd
from keras.models import load_model

entrada = pd.read_csv('./data/input.csv')

model = load_model('classificador_xor.h5')

previsao = model.predict(entrada)
previsao = (previsao > 0.5)
print(previsao)

'''
Realizando avaliação da rede neural
'''
previsores = pd.read_csv('./data/input.csv')
classe = pd.read_csv('./data/output.csv')
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['binary_accuracy'])
resultado = model.evaluate(previsores, classe)
print(resultado)
