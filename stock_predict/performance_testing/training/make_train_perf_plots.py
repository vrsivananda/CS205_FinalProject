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
plt.ylabel('Speedup')
plt.xticks(n_nodes)
plt.xlabel('Number of Nodes')
plt.legend()
plt.savefig('speedup_multi_node.png')


# Plot 2: Multi-GPU speedup

with open('multi_gpu.txt') as json_file:
    multi_gpu = json.load(json_file)
    
n_gpus = [1,2]
speedup = [1, np.mean(multi_node['1']['time_per_epoch'])/ np.mean(multi_gpu['bs_32']['time_per_epoch'])]

plt.figure()
plt.plot(n_gpus, speedup, label='Actual speedup')
plt.plot(n_gpus, n_gpus, label='Theoretical speedup', linestyle='--')
plt.title('Speedup with Increasing GPUs in a Single Node')
plt.ylabel('Speedup')
plt.xticks(n_gpus)
plt.xlabel('Number of GPUs')
plt.legend()
plt.savefig('speedup_multi_gpu.png')

# Plot 3: Batch size (multi node)

with open('multi_node.txt') as json_file:
    multi_node = json.load(json_file)

steps_per_epoch = [24226, 12113, 6056, 3028]
batch_sizes = [32, 64, 96, 128]
mean_time_per_epoch = []

for k, nodes_dict in multi_node.items():
    mean_time_per_epoch.append(np.mean(nodes_dict['time_per_epoch'])*1000)
    
time_per_step = np.array(mean_time_per_epoch)/np.array(steps_per_epoch)


fig, ax1 = plt.subplots()
color = 'tab:red'

ax1.plot(batch_sizes, time_per_step, label='', color=color)
ax1.set_ylabel('Time per step (ms)', color=color)
ax1.tick_params(axis='y', labelcolor=color)
#ax1.set_ylim(ymin=0)
ax1.set_xticks(batch_sizes)
ax1.set_xlabel('Effective Batch Size')

ax1.set_title('Multi Node')

ax2 = ax1.twinx()
color = 'tab:blue'
ax2.plot(batch_sizes, np.array(mean_time_per_epoch)/1000, label='')
ax2.set_ylabel('Time per epoch (s)', color=color)
#ax2.set_ylim(ymin=0)
ax2.tick_params(axis='y', labelcolor=color)

fig.tight_layout()
fig.savefig('time_vs_batchsize_multi_node.png')


# Plot 4: Batch size (single node)

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
#ax1.set_ylim(ymin=0)
ax1.set_xticks(batch_sizes)
ax1.set_xlabel('Effective Batch Size')

ax1.set_title('Single GPU')

ax2 = ax1.twinx()
color = 'tab:blue'
ax2.plot(batch_sizes, np.array(mean_time_per_epoch)/1000, label='')
ax2.set_ylabel('Time per epoch (s)', color=color)
#ax2.set_ylim(ymin=0)
ax2.tick_params(axis='y', labelcolor=color)

fig.tight_layout()
fig.savefig('time_vs_batchsize_single_gpu.png')


# Plot 5: Batch size (multi GPU)

with open('multi_gpu.txt') as json_file:
    multi_gpu = json.load(json_file)

steps_per_epoch = [24226, 12113, 6056, 3028]
batch_sizes_multi_gpu = [64, 128, 256, 512]
mean_time_per_epoch = []

for k, bs_dict in multi_gpu.items():
    mean_time_per_epoch.append(np.mean(bs_dict['time_per_epoch'])*1000)
    
time_per_step = np.array(mean_time_per_epoch)/np.array(steps_per_epoch)


fig, ax1 = plt.subplots()
color = 'tab:red'

ax1.plot(batch_sizes_multi_gpu[:3], time_per_step[:3], label='', color=color)
ax1.set_title('')
ax1.set_ylabel('Time per step (ms)', color=color)
ax1.tick_params(axis='y', labelcolor=color)
#ax1.set_ylim(ymin=0)
ax1.set_xticks(batch_sizes_multi_gpu[:3])
ax1.set_xlabel('Effective Batch Size')

ax1.set_title('Multi GPU')

ax2 = ax1.twinx()
color = 'tab:blue'
ax2.plot(batch_sizes_multi_gpu[:3], (np.array(mean_time_per_epoch)/1000)[:3], label='')
ax2.set_ylabel('Time per epoch (s)', color=color)
#ax2.set_ylim(ymin=0)
ax2.tick_params(axis='y', labelcolor=color)

fig.tight_layout()
fig.savefig('time_vs_batchsize_multi_gpu.png')


# Plot 6: Testing accuracy (multi node & multi GPU)

with open('multi_gpu.txt') as json_file:
    multi_gpu = json.load(json_file)

with open('multi_node.txt') as json_file:
    multi_node = json.load(json_file)


batch_sizes_multi_gpu = [64, 128, 256, 512]
batch_sizes_multi_node = [32, 64, 96, 128]
test_loss_multi_gpu = []
test_loss_multi_node = []

for k, bs_dict in multi_gpu.items():
    test_loss_multi_gpu.append(bs_dict['test_loss'])

for k, node_dict in multi_node.items():
    test_loss_multi_node.append(node_dict['test_loss'])
    
   
print(batch_sizes_multi_node[1:])
print(np.sqrt(test_loss_multi_node)[1:])
print(batch_sizes_multi_gpu[:2])
print(np.sqrt(test_loss_multi_gpu)[:2])


plt.figure()
plt.plot(batch_sizes_multi_gpu[:2], np.sqrt(test_loss_multi_gpu)[:2], label='Multi GPU')
plt.plot(batch_sizes_multi_node[1:], np.sqrt(test_loss_multi_node)[1:], label='Multi Node')
plt.title('Testing Loss for Multi Node and Multi GPU')
plt.ylabel('Test Loss (RMSE)')
plt.ylim(ymin=0)
plt.xticks(batch_sizes_multi_node[1:])
plt.xlabel('Effective Batch Size')
plt.legend()
plt.savefig('test_loss_multi_node_multi_gpu.png')


# Plot 7: Speedup with batch size (single & multi GPU)

with open('multi_gpu.txt') as json_file:
    multi_gpu = json.load(json_file)

with open('multi_node.txt') as json_file:
    single_node = json.load(json_file)

batch_sizes_multi_gpu = [64, 128, 256, 512]
batch_sizes_multi_node = [32, 64, 96, 128]
total_time_multi_gpu = []
total_time_multi_node = []

for k, bs_dict in multi_gpu.items():
    total_time_multi_gpu.append(np.sum(bs_dict['time_per_epoch']))

for k, bs_dict in multi_node.items():
    total_time_multi_node.append(np.sum(bs_dict['time_per_epoch']))

speedup_multi_gpu = [total_time_multi_gpu[0]/x for x in total_time_multi_gpu]
speedup_multi_node = [total_time_multi_node[1]/x for x in total_time_multi_node]
    
print(total_time_multi_gpu)
print(speedup_multi_node)

plt.figure()
plt.plot(batch_sizes_multi_gpu[:2], speedup_multi_gpu[:2], label='Multi GPU')
plt.plot(batch_sizes_multi_node[1:], speedup_multi_node[1:], label='Multi Node')
plt.title('Speedup with Increasing Effective Batch Sizes')
plt.ylabel('Speedup')
plt.xlabel('Effective Batch Size')
plt.xticks(batch_sizes_multi_node[1:])
plt.legend()
plt.savefig('speedup_multi_node_multi_gpu.png')
