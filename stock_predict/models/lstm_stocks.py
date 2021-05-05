import tensorflow as tf
import numpy as np
import horovod.keras as hvd

# Horovod: initialize Horovod.
hvd.init()

gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)
if gpus:
    print('gpus')
    tf.config.experimental.set_visible_devices(gpus[hvd.local_rank()], 'GPU')

data = np.load('training_data.npz')

x_train = tf.convert_to_tensor(data['x_train'])
y_train = tf.convert_to_tensor(data['y_train'])


xmask = np.max(np.isnan(x_train).astype(int), axis=(1,2)) == 0
x_train = x_train[xmask]
y_train = y_train[xmask]

ymask = np.isnan(y_train) == False
x_train = x_train[ymask]
y_train = y_train[ymask]

# Create training, test sets
train_size = 0.75
count = int(len(y_train)*(1-train_size))
idx = np.random.choice(len(y_train), count, replace=False)
nonidx = np.array([i for i in range(len(y_train)) if i not in idx])
#x_test = x_train[idx]
#y_test = y_train[idx]
#x_train = x_train[] 



mod = tf.keras.Sequential([
    tf.keras.layers.LSTM(16, return_sequences=True),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.LSTM(16),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(32),
    tf.keras.layers.Dense(1)
])

# Horovod: adjust learning rate based on number of GPUs.
opt = tf.keras.optimizers.Adam(1e-3) #.Adadelta(1.0 * hvd.size())

# Horovod: add Horovod Distributed Optimizer.
opt = hvd.DistributedOptimizer(opt)

mod.compile(optimizer=opt, loss='mse')
mod.summary()

callbacks = [
    # Horovod: broadcast initial variable states from rank 0 to all other processes.
    # This is necessary to ensure consistent initialization of all workers when
    # training is started with random weights or restored from a checkpoint.
    hvd.callbacks.BroadcastGlobalVariablesCallback(0),
]

# Horovod: save checkpoints only on worker 0 to prevent other workers from corrupting them.
if hvd.rank() == 0:
    callbacks.append(tf.keras.callbacks.ModelCheckpoint('./checkpoint-{epoch}.h5'))


h = mod.fit(x=x_train, y=y_train,
            epochs=1,
            batch_size=32,
            validation_split=0.25,
            verbose=1 if hvd.rank() == 0 else 0)
