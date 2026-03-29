import os
import yfinance as yf
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from scipy.signal import stft
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, Flatten, Dense

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

print("Downloading data...")
data = yf.download("RELIANCE.NS", start="2020-01-01", end="2024-01-01")
print("Download complete")

close_data = data['Close']

plt.figure()
plt.plot(close_data)
plt.title("Stock Price")
plt.show(block=False)
plt.pause(2)
plt.close()

scaler = MinMaxScaler()
normalized = scaler.fit_transform(close_data.values.reshape(-1,1))
signal = normalized.flatten()

signal = signal[:500]

print("Preprocessing done")

f, t, Zxx = stft(signal, nperseg=32, noverlap=16)
spectrogram = np.abs(Zxx)**2

plt.figure()
plt.pcolormesh(t, f, spectrogram)
plt.title("Spectrogram")
plt.ylabel("Frequency")
plt.xlabel("Time")
plt.show(block=False)
plt.pause(2)
plt.close()

window_size = 64
X = []
y = []

print("Starting STFT loop...")

for i in range(len(signal) - window_size - 1):
    if i % 100 == 0:
        print("Processing:", i)

    segment = signal[i:i+window_size]
    f, t, Zxx = stft(segment, nperseg=16, noverlap=8)
    spec = np.abs(Zxx)**2

    X.append(spec)
    y.append(signal[i+window_size])

X = np.array(X)
y = np.array(y)

X = X.reshape(X.shape[0], X.shape[1], X.shape[2], 1)

print("Data prepared")
print("X shape:", X.shape)
print("y shape:", y.shape)

model = Sequential()
model.add(Conv2D(16, (3,3), activation='relu', input_shape=X.shape[1:]))
model.add(Flatten())
model.add(Dense(32, activation='relu'))
model.add(Dense(1))

model.compile(optimizer='adam', loss='mse')

print("Starting training...")
model.fit(X, y, epochs=3)

prediction = model.predict(X[-1].reshape(1, X.shape[1], X.shape[2], 1))

print("Prediction:", prediction)