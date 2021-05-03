import tensorflow as tf
import numpy as np

data = np.load('training_data.npz')

x_train = tf.convert_to_tensor(data['x_train'])
y_train = tf.convert_to_tensor(data['y_train'])


xmask = np.max(np.isnan(x_train).astype(int), axis=(1,2)) == 0
x_train = x_train[xmask]
y_train = y_train[xmask]

ymask = np.isnan(y_train) == False
x_train = x_train[ymask]
y_train = y_train[ymask]

mod = tf.keras.Sequential([
    tf.keras.layers.LSTM(8, return_sequences=True),
    tf.keras.layers.LSTM(8),
    tf.keras.layers.Dense(16),
    tf.keras.layers.Dense(1)
])

mod.compile(optimizer=tf.keras.optimizers.Adam(1e-3), loss='mse')

h = mod.fit(x=x_train, y=y_train, epochs=10, batch_size=64, validation_split=0.25, verbose=1)

