import numpy as np
from numpy import genfromtxt
import matplotlib.pyplot as plt

# Load data
val_losses = genfromtxt('val_loss_single_node.csv', delimiter=',')
epoch_times = genfromtxt('epoch_time_single_node.csv', delimiter=',')
batch_sizes = [2**x for x in range(5,12)]


# Remove batch sizes that are too large so we can plot clearly
n_batch_sizes = 4
val_losses = val_losses[:n_batch_sizes]
epoch_times = epoch_times[:n_batch_sizes]
batch_sizes = batch_sizes[:n_batch_sizes]

epochs = range(len(val_losses[0]))

#----------------------------------

# Plot the loss vs epoch for each batch size

plt.figure
for i, batch_size in enumerate(batch_sizes):
    plt.plot(epochs, val_losses[i], label=f'batch_size = {batch_size}')

plt.title('Validation loss with different batch sizes\n by epoch')
plt.xlabel('epochs')
plt.xticks(epochs)
plt.ylabel('validation loss')
plt.yticks(range(0,int(np.max(np.max(val_losses))),2000))
plt.legend()
plt.savefig('val_loss_single_node.png')

#----------------------------------

# Plot the time that it takes to get to a loss of around 3000 for each batch size

total_time = [np.cumsum(x) for x in epoch_times]

plt.figure()
for i, batch_size in enumerate(batch_sizes):
    plt.plot(total_time[i], val_losses[i], label=f'batch_size = {batch_size}')

plt.title('Validation loss with different batch sizes\n by real time (s)')
plt.xlabel('time (s)')
plt.xticks(range(0,300,50))
plt.xlim([0,300])
plt.ylabel('validation loss')
plt.yticks(range(0,int(np.max(np.max(val_losses))),2000))
plt.legend()
plt.savefig('epoch_time_single_node.png')

