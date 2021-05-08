# Instructions

#### Single Node Mode

1. Spin up g4dn.xlarge AWS instance (or AWS g3s.8xlarge), using Ubuntu 18.04. Log in with appropriate AWS credentials.
2. Expand the volume to 128GB
   1. This can be done through `Instances > [Select Instance ID] > Storage > [Select only Volume ID] > Actions > Modify Volume > Request 128GB`
3. SSH into the instance, and transfer `horovod_all.sh`, `lstm_stocks.py`, `keras_mnist.py`. Note that if not using AWS CLI to transfer in training data, that can be done here as well.
4. Expand volume through the following steps:
   1. `sudo growpart /dev/nvme0n1 1` or `sudo growpart /dev/xvda 1`
   2. `sudo resize2fs /dev/nvme0n1p1` or `sudo resize2fs /dev/xvda1`
   3. Confirm memory available via `df -h`. Resized part should be $> 128$ GB.
5. Run the following commands to ensure shell script compatible with Linux:

```bash
sudo chmod +x horovod_all.sh
sed -i -e 's/\r$//' horovod_all.sh
```

6. Execute ```./horovod_all.sh```
7. Verify that TF, MPI, and NCCL are available from the output of `horovodrun --check-build` (this will run automatically in the shell, but provides a visual confirmation of installation).
8. Verify `keras_mnist.py` runs. This is a short proof of concept that the installation is correct, though the output can be quite verbose.
   1. With GPU enabled, first epoch should be approximately 6-10 seconds, second epoch approximately 3-6 seconds.
   2. GPU can also be confirmed by separately SSH-ing into the instance, and running `nvidia-smi` while script is running to verify GPU is working.
9. If training data uploaded, execute the following commands:

```bash
export PATH="/home/ubuntu/.local/bin:$PATH"
horovodrun -np 1 -H localhost:1 python3.8 lstm_stocks.py
```

10. If training data is not uploaded, follow the instructions below to retrieve from AWS. Note: AWS CLI must be configured.

    ```bash
    aws configure
    aws s3 ls
    ```

    ```bash
    aws s3 cp s3://cs205-stream-stock-predict/training_data.npz .
    ```

    Then return to Step 9.

#### Multi-Node Mode

The instructions for the multi-node mode are similar to the single GPU instance. First, however, we configure the MPI implementation on both nodes. This follows the infrastructure guide [here](https://harvard-iacs.github.io/2021-CS205/labs/I7_2/I7_2.pdf) at most steps, but is reproduced below for clarity.

1. Spin up VPC Cluster, initializing VPC and subnet. Request the number of instances desired (i.e number of GPUs desired).
2. Add Internet Gateway and attach to VPC. Use a destination of `0.0.0.0/0` and a target of the VPC.
3. Decide on which node is the manager node and which are workers. We have found it helpful to write the public & private IPs down (the private IPs won't change) to keep track of how things are working. Add the private IPs of each node to the `/etc/hosts` file, naming `master` and `node1`, `node2`, etc.
```shell
sudo vi /etc/hosts
```
4. Configure MPI User Accounts. Note that the following instructions come from AWS Infrastructure Guide #7, Harvard CS205, Spring 2021. `master$` indicates the master node and `nodeN$` indicates a command to be run on each worker node.

```shell
master$ sudo adduser mpiuser
master$ sudo adduser mpiuser sudo
nodeN$ sudo adduser mpiuser
nodeN$ sudo adduser mpiuser sudo
```

- Add SSH to all nodes

```shell
sudo vi /etc/ssh/sshd_config
```

and change lines 55-56 to be:

```shell
# Change to no to disable tunnelled clear text passwords
PasswordAuthentication yes
```

- Restart service on all nodes

```shell
master$ sudo service ssh restart
nodeN$ sudo service ssh restart
```

- Log into all N nodes from master:

```shell
master$ su - mpiuser
mpiuser@master$ ssh mpiuser@nodeN
```

- Note that the previous step can be repeated on each node if desired
- Generate SSH pairs on each node, copying to all other nodes (i.e. the second command must be repeated for all N nodes). Accept the default values.

```shell
mpiuser@master$ ssh-keygen
mpiuser@master$ ssh-copy-id mpiuser@nodeN
```

- Add NFS server and make cloud directory

```shell
master$ sudo apt-get install nfs-kernel-server
master$ su - mpiuser
mpiuser@master$ mkdir cloud
```

- on Master (not `mpiuser@master`), add `/home/mpiuser/cloud *(rw,sync,no_root_squash,no_subtree_check)` to the file `/etc/exports` using: `master$ sudo vi /etc/exports`. Then run:

```shell
master$ sudo exportfs -a
master$ sudo service nfs-kernel-server restart
```

- Configure NFS client on each node:

```shell
nodeN$ sudo apt-get install nfs-common
nodeN$ su - mpiuser
mpiuser@nodeN$ mkdir cloud
```

- Add inbound rule to security group, allowing Type = `NFS` and Source = `0.0.0.0/0`
- Mount shared directory in each node, confirm file is in system:

```shell
nodeN$ sudo mount -t nfs master:/home/mpiuser/cloud /home/mpiuser/cloud
nodeN$ df -h
```

2. Now we differ from the infrastructure guide. To install all relevant packages, transfer `horovod_all.sh` to each node.
3. Expand volume on each node. See Steps 2 & 4 of Single GPU mode, repeat on each instance and confirm memory available.
4. Execute the following on each node (& master). Given the time to run this, we *highly* suggest doing this in parallel:

```shell
master$ sudo chmod +x horovod_all.sh
master$ sed -i -e 's/\r$//' horovod_all.sh
master$ ./horovod_all.sh
```

5. As with above, verify that all information is correct on each node for Keras MNIST.
6. Run `export PATH="/home/ubuntu/.local/bin:$PATH"` in each node.
7. Open an MPI port where Type = `All TCP` in Inbound Security Rules, with Source = `0.0.0.0/0`. Allow the port range to be any port.
8. Once the above has finished on each node, then on master node, upload `horovod_mpiuser_master.sh` and `keras_mnist.py` to the *mpiuser* user of the master node, and execute the following command:

```shell
mpiuser@master$ sudo chmod +x horovod_mpiuser_master.sh
mpiuser@master$ sed -i -e 's/\r$//' horovod_mpiuser_master.sh
mpiuser@master$ ./horovod_mpiuser_master.sh
```

This should run OpenMPI commands, producing basic output of 'Hello from process N', confirming the correct installation of OpenMPI.

9. On each node, upload `horovod_mpiuser_nodes.sh` and `keras_mnist.py`, and run the following command:

```shell
mpiuser@nodeN$ sudo chmod +x horovod_mpiuser_nodes.sh
mpiuser@nodeN$ sed -i -e 's/\r$//' horovod_mpiuser_nodes.sh
mpiuser@nodeN$ ./horovod_mpiuser_nodes.sh
```

10. Execute the following commands in each node:

```shell
export PATH="/home/ubuntu/.local/bin:$PATH"
export PATH="/home/mpiuser/.local/bin:$PATH"
```

11. Follow step 10 of single GPU mode to retrieve training data, and copy to `cloud`.
12. To run, execute the following command in the `cloud` directory. Note that each additional node will be another host, and depends on naming convention, and that the Python script and `training_data.npz` must both be in `cloud`. Two and four nodes are shown as examples.

```shell
horovodrun -np 2 -H master:1,node1:1 python3.8 lstm_stocks.py
horovodrun -np 2 -H master:1,node1:1,node2:1,node3:1 python3.8 lstm_stocks.py
```

13. If running on many GPUs on the same node, the following commands implements it correctly:

```shell
horovodrun -np 2 -H localhost:2 python3.8 lstm_stocks.py
horovodrun -np 8 -H master:2,node1:2,node2:2,node3:2 python3.8 lstm_stocks.py
```
