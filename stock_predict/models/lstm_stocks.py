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

x_train = data['x_train']
y_train = data['y_train']

# Subset data to be non-missing
xmask = np.max(np.isnan(x_train).astype(int), axis=(1,2)) == 0
x_train = x_train[xmask]
y_train = y_train[xmask]

ymask = np.isnan(y_train) == False
x_train = x_train[ymask]
y_train = y_train[ymask]

# Standardize data to improve fit
x_train_min = x_train.min(axis=0)
x_train_max = x_train.max(axis=0)
y_train_min = y_train.min()
y_train_max = y_train.max()

x_train = (x_train - x_train_min)/(x_train_max - x_train_min)
y_train = (y_train - y_train_min)/(y_train_max - y_train_min)

# Create training, test sets
train_size = 0.9
count = int(len(y_train)*(1-train_size))
idx = np.random.choice(len(y_train), count, replace=False)
nonidx = np.setdiff1d(np.arange(len(y_train)), idx)

x_test = x_train[idx]
y_test = y_train[idx]
x_train = x_train[nonidx]
y_train = y_train[nonidx]

x_train = tf.convert_to_tensor(x_train)
y_train = tf.convert_to_tensor(y_train)
x_test = tf.convert_to_tensor(x_test)
y_test = tf.convert_to_tensor(y_test)


## Create model architecture
mod = tf.keras.Sequential([
    tf.keras.layers.LSTM(50),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(32, activation='relu'),
    tf.keras.layers.Dense(1, activation='relu')
])

# Horovod: adjust learning rate based on number of GPUs.
opt = tf.keras.optimizers.Adam(1e-4 * hvd.size()) #.Adadelta(1.0 * hvd.size())

# Horovod: add Horovod Distributed Optimizer.
opt = hvd.DistributedOptimizer(opt)

mod.compile(optimizer=opt, loss='mse')
#mod.summary()

callbacks = [
    # Horovod: broadcast initial variable states from rank 0 to all other processes.
    # This is necessary to ensure consistent initialization of all workers when
    # training is started with random weights or restored from a checkpoint.
    hvd.callbacks.BroadcastGlobalVariablesCallback(0),
]

# Horovod: save checkpoints only on worker 0 to prevent other workers from corrupting them.
if hvd.rank() == 0:
    callbacks.append(tf.keras.callbacks.ModelCheckpoint('./checkpoint-{epoch}.h5'))


## ADD MODEL HYPERPARAMETERS
BATCH_SIZE = 32
EPOCHS = 20
VALIDATION_SPLIT = 0.2
STEPS_PER_EPOCH = len(y_train)*(1-VALIDATION_SPLIT) // (BATCH_SIZE * hvd.size())

if hvd.rank() == 0:
    print(f'\n\n\n\n\n ######### batch_size: {BATCH_SIZE} #########')
    print(f'Horovod Size: {hvd.size()}')
    print(f'Train size: {int(len(y_train)*(1-VALIDATION_SPLIT))}')
    print(f'Steps per epoch: {int(STEPS_PER_EPOCH)}')
    print(f'Epochs: {int(EPOCHS)}')

h = mod.fit(x=x_train, y=y_train,
            epochs=EPOCHS,
            batch_size=BATCH_SIZE,
            steps_per_epoch=STEPS_PER_EPOCH,
            validation_split=VALIDATION_SPLIT,
            verbose=1 if hvd.rank() == 0 else 0)

# Evaluate model to ensure accuracy
if hvd.rank() == 0:
    mod.evaluate(x=x_test, y=y_test, batch_size=BATCH_SIZE, verbose=1)
    # Save out model for transfer to prediction phase
    mod.save(f'trained_lstm_mod_{hvd.size()}_{BATCH_SIZE}.h5')
    for key, val in h.history.items():
        print(f'{key}: {val}')
    print('^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^\n\n\n\n\n')
