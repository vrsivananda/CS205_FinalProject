# Install pip
sudo apt install -y python3-pip
#pip3 install --upgrade pip
python3.8 -m pip3 install --upgrade pip
# ^ Stuff on pip needs to be installed like this instead for Python3.8 to find them

# Install tensorflow
#python3.8 -m pip3 install --upgrade tensorflow
pip3.8 install --upgrade tensorflow
python3.8 -c "import tensorflow as tf;print(tf.reduce_sum(tf.random.normal([1000, 1000])))"

# Install horovod
HOROVOD_WITH_TENSORFLOW=1 HOROVOD_GPU_OPERATIONS=NCCL HOROVOD_WITH_MPI=1 pip3.8 install --no-cache-dir horovod[tensorflow,keras]
export PATH="/home/ubuntu/.local/bin:$PATH"
export PATH="/home/mpiuser/.local/bin:$PATH"
horovodrun --check-build
horovodrun -np 1 -H localhost:1 python3.8 keras_mnist.py

## Test MPI Set-Up
wget https://harvard-iacs.github.io/2021-CS205/labs/I7_2/mpi_sc.c
mpicc mpi_sc.c -o mpi_sc
mpirun -np 2 ./mpi_sc

wget https://harvard-iacs.github.io/2021-CS205/labs/I7_2/mpi_sch.c
mpicc mpi_sch.c -o mpi_sch
cp mpi_sch cloud
cd cloud
mpirun -np 4 -H master,node1,node2,node3 ./mpi_sch

