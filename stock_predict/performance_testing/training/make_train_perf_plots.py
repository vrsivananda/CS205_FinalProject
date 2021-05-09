import json
import numpy as np
import matplotlib.pyplot as plt

# Plot 1: Multi-node speedup

with open('multi_node.txt') as json_file:
    multi_node = json.load(json_file)
    
n_nodes = [1,2,3,4]
avg_time_per_epoch = []
speedup = []
for k, node in multi_node.items():
    avg_time_per_epoch.append(np.mean(node['time_per_epoch']))
    speedup.append(avg_time_per_epoch[0]/avg_time_per_epoch[-1])
    
print(speedup)
print(n_nodes)
    
plt.figure()
plt.plot(n_nodes, speedup, label='Actual speedup')
plt.plot(n_nodes, n_nodes, label='Theoretical speedup', linestyle='--')
plt.title('Speedup with increasing nodes')
plt.ylabel('speedup')
plt.xticks(n_nodes)
plt.xlabel('number of nodes')
plt.legend()
plt.savefig('multi_node_speedup.png')


# Plot 2: Multi-GPU speedup

with open('multi_gpu.txt') as json_file:
    multi_gpu = json.load(json_file)
    
n_gpus = [1,2]
speedup = [1, np.mean(multi_node['1']['time_per_epoch'])/ np.mean(multi_gpu['bs_32']['time_per_epoch'])]

plt.figure()
plt.plot(n_gpus, speedup, label='Actual speedup')
plt.plot(n_gpus, n_gpus, label='Theoretical speedup', linestyle='--')
plt.title('Speedup with increasing GPUs in a single node')
plt.ylabel('Speedup')
plt.xticks(n_gpus)
plt.xlabel('Number of GPUs')
plt.legend()
plt.savefig('multi_gpu_speedup.png')


# Plot 3: Batch size (single node)

with open('single_node.txt') as json_file:
    single_node = json.load(json_file)

steps_per_epoch = [24226, 12113, 6056, 3028]
batch_sizes = [32, 64, 96, 128]
mean_time_per_epoch = []

for k, bs_dict in single_node.items():
    mean_time_per_epoch.append(np.mean(bs_dict['time_per_epoch'])*1000)
    
time_per_step = np.array(mean_time_per_epoch)/np.array(steps_per_epoch)


fig, ax1 = plt.subplots()
color = 'tab:red'

ax1.plot(batch_sizes, time_per_step, label='', color=color)
ax1.set_ylabel('Time per step (ms)', color=color)
ax1.tick_params(axis='y', labelcolor=color)
ax1.set_xticks(batch_sizes)
ax1.set_xlabel('Effective Batch Size')

ax1.set_title('Single Node')

ax2 = ax1.twinx()
color = 'tab:blue'
ax2.plot(batch_sizes, np.array(mean_time_per_epoch)/1000, label='')
ax2.set_ylabel('Time per epoch (s)', color=color)
ax2.tick_params(axis='y', labelcolor=color)

fig.tight_layout()
fig.savefig('time_vs_batchsize_single_node.png')


# Plot 4: Batch size (multi node)

with open('multi_node.txt') as json_file:
    multi_node = json.load(json_file)

steps_per_epoch = [24226, 12113, 6056, 3028]
batch_sizes = [32, 64, 96, 128]
mean_time_per_epoch = []

for k, bs_dict in multi_node.items():
    mean_time_per_epoch.append(np.mean(bs_dict['time_per_epoch'])*1000)
    
time_per_step = np.array(mean_time_per_epoch)/np.array(steps_per_epoch)


fig, ax1 = plt.subplots()
color = 'tab:red'

ax1.plot(batch_sizes, time_per_step, label='', color=color)
ax1.set_ylabel('Time per step (ms)', color=color)
ax1.tick_params(axis='y', labelcolor=color)
ax1.set_xticks(batch_sizes)
ax1.set_xlabel('Effective Batch Size')

ax1.set_title('Multi Node')

ax2 = ax1.twinx()
color = 'tab:blue'
ax2.plot(batch_sizes, np.array(mean_time_per_epoch)/1000, label='')
ax2.set_ylabel('Time per epoch (s)', color=color)
ax2.tick_params(axis='y', labelcolor=color)

fig.tight_layout()
fig.savefig('time_vs_batchsize_multi_node.png')


# Plot 5: Batch size (multi GPU)

with open('multi_gpu.txt') as json_file:
    multi_gpu = json.load(json_file)

steps_per_epoch = [24226, 12113, 6056, 3028]
batch_sizes = [32, 64, 96, 128]
mean_time_per_epoch = []

for k, bs_dict in multi_gpu.items():
    mean_time_per_epoch.append(np.mean(bs_dict['time_per_epoch'])*1000)
    
time_per_step = np.array(mean_time_per_epoch)/np.array(steps_per_epoch)


fig, ax1 = plt.subplots()
color = 'tab:red'

ax1.plot(batch_sizes, time_per_step, label='', color=color)
ax1.set_title('')
ax1.set_ylabel('Time per step (ms)', color=color)
ax1.tick_params(axis='y', labelcolor=color)
ax1.set_xticks(batch_sizes)
ax1.set_xlabel('Effective Batch Size')

ax1.set_title('Multi GPU')

ax2 = ax1.twinx()
color = 'tab:blue'
ax2.plot(batch_sizes, np.array(mean_time_per_epoch)/1000, label='')
ax2.set_ylabel('Time per epoch (s)', color=color)
ax2.tick_params(axis='y', labelcolor=color)

fig.tight_layout()
fig.savefig('time_vs_batchsize_multi_gpu.png')
