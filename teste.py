import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
import matplotlib.pyplot as plt

file_path = 'WDOohlc.csv'

df = pd.read_csv(file_path)
def load_data(file_path):
    # Lê o CSV corretamente com tabulação
    df = pd.read_csv(file_path, sep='\t', engine='python', encoding='utf-8')
    
    # Verifique os nomes das colunas
    print("Colunas do CSV:", df.columns)
    
    # Seleciona a coluna <CLOSE>
    data = df['<CLOSE>'].values.reshape(-1, 1)
    
    # Normaliza os dados entre 0 e 1
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(data)
    
    return scaled_data, scaler

data, scaler = load_data('WDOohlc.csv')
print(data[:5])  # Mostra os primeiros 5 preços normalizados
