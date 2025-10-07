import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error


# -------------------------
# Load and preprocess the data
# -------------------------
def load_data(file_path):
    # Lê o CSV com tabulação corretamente
    df = pd.read_csv(file_path, sep='\t', engine='python', encoding='utf-8')
    
    # Confirma os nomes das colunas
    print("Colunas do CSV:", df.columns)
    
    # Seleciona a coluna <CLOSE>
    data = df['<CLOSE>'].values.reshape(-1, 1)
    
    # Normaliza os dados entre 0 e 1
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(data)
    
    return scaled_data, scaler

def create_dataset(data, time_step):
    x, y = [], []
    for i in range(len(data) - time_step - 1):
        x.append(data[i:(i + time_step), 0])
        y.append(data[i + time_step, 0])
    return np.array(x), np.array(y)

# -------------------------
# Hyperparameters
# -------------------------
time_step = 60  # Number of time steps to look back
epochs = 100
batch_size = 32

# -------------------------
# Load data
# -------------------------
file_path = 'WDOohlc.csv'  # Certifique-se de que o arquivo esteja na mesma pasta do script
data, scaler = load_data(file_path)

# -------------------------
# Create train and test sets
# -------------------------
train_size = int(len(data) * 0.8)
train_data = data[0:train_size, :]
test_data = data[train_size - time_step:, :]

x_train, y_train = create_dataset(train_data, time_step)
x_test, y_test = create_dataset(test_data, time_step)

# Reshape input para [samples, time steps, features]
x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))
x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))

# -------------------------
# Build LSTM model
# -------------------------
model = Sequential()
model.add(LSTM(units=50, return_sequences=True, input_shape=(x_train.shape[1], 1)))
model.add(LSTM(units=50, return_sequences=False))
model.add(Dense(units=25))
model.add(Dense(units=1))

# Compile the model
model.compile(optimizer='adam', loss='mean_squared_error')

# Train the model
model.fit(x_train, y_train, epochs=epochs, batch_size=batch_size, verbose=1)

# -------------------------
# Make predictions
# -------------------------
predictions = model.predict(x_test)
predictions = scaler.inverse_transform(predictions)
y_test_scaled = scaler.inverse_transform(y_test.reshape(-1, 1))

# -------------------------
# Visualize predictions vs actual values
# -------------------------
plt.figure(figsize=(12, 6))
plt.plot(y_test_scaled, label='Actual Stock Price', color='blue')
plt.plot(predictions, label='Predicted Stock Price', color='red')
plt.title('LSTM Stock Price Prediction')
plt.xlabel('Time')
plt.ylabel('Stock Price')
plt.legend()
plt.show()

mae = mean_absolute_error(y_test_scaled, predictions)
print(f"Mean Absolute Error (MAE): {mae}")

rmse = np.sqrt(np.mean((y_test_scaled - predictions) ** 2))
print(f"Root Mean Squared Error (RMSE): {rmse}")


