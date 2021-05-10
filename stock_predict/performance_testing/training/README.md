## Performance Testing for Training Phase

This directory contains the performance testing results for the training phase.

#### Scripts:

1. `make_train_perf_data.py`: This contains the code that generates the `.txt` files which contain the data in JSON format.
2. `make_train_perf_plots.py`: This contains code to read the data from the `.txt` files and generate the graphs.

#### Data files:
1. `single_node.txt`: This file contains the data from a single g3s.xlarge AWS instance, scaled across different batch sizes.
2. `multi_node.txt`: This file contains the data from 1-4 g3s.xlarge AWS instances that form a cluster
3. `multi_gpu.txt`: This file contains the data from a single g3.8xlarge AWS instance, which has 2 GPUs.
