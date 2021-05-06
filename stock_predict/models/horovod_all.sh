
# Run this in the terminal to remove the spurious CR characters (added from Windows machine)
#sed -i -e 's/\r$//' horovod_all.sh

# Increase volume
sudo apt install -y awscli
df -h
lsblk
#sudo growpart /dev/xvda 1
#sudo resize2fs /dev/xvda1

# Install Python3.8
sudo apt update
sudo apt install software-properties-common
sudo add-apt-repository -y ppa:deadsnakes/ppa
sudo apt install -y python3.8
python3.8 --version

# Install cmake
sudo apt-add-repository universe
sudo apt-get update
sudo apt-get install -y cmake


#Install CUDA
lspci | grep -i nvidia
sudo apt-get install build-essential
gcc --version
wget http://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/cuda-repo-ubuntu1804_10.0.130-1_amd64.deb
sudo apt-key adv --fetch-keys http://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/7fa2af80.pub
sudo dpkg -i cuda-repo-ubuntu1804_10.0.130-1_amd64.deb
sudo apt-get update
sudo apt-get install -y cuda
nvidia-smi
sudo nvidia-smi -pm 1
sudo nvidia-smi --auto-boost-default=0
sudo nvidia-smi -ac 2505,875


### FROM TENSORFLOW GPU DOCUMENTATION
# Add NVIDIA package repositories
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/cuda-ubuntu1804.pin
sudo mv cuda-ubuntu1804.pin /etc/apt/preferences.d/cuda-repository-pin-600
sudo apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/7fa2af80.pub
sudo add-apt-repository "deb https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/ /"
sudo apt-get update

wget http://developer.download.nvidia.com/compute/machine-learning/repos/ubuntu1804/x86_64/nvidia-machine-learning-repo-ubuntu1804_1.0.0-1_amd64.deb

sudo apt install ./nvidia-machine-learning-repo-ubuntu1804_1.0.0-1_amd64.deb
sudo apt-get update

wget https://developer.download.nvidia.com/compute/machine-learning/repos/ubuntu1804/x86_64/libnvinfer7_7.1.3-1+cuda11.0_amd64.deb
sudo apt install ./libnvinfer7_7.1.3-1+cuda11.0_amd64.deb
sudo apt-get update

# Install development and runtime libraries (~4GB)
sudo apt-get install --no-install-recommends \
    cuda-11-0 \
    libcudnn8=8.0.4.30-1+cuda11.0  \
    libcudnn8-dev=8.0.4.30-1+cuda11.0

# Install NCCL2
sudo apt install libnccl2=2.9.6-1+cuda11.0 libnccl-dev=2.9.6-1+cuda11.0

# Install pip
sudo apt install -y python3-pip
pip3 install --upgrade pip
# ^ Stuff on pip needs to be installed like this instead for Python3.8 to find them

# Install tensorflow
#python3.8 -m pip install --upgrade tensorflow
pip3 install --upgrade tensorflow
python3.8 -c "import tensorflow as tf;print(tf.reduce_sum(tf.random.normal([1000, 1000])))"

# Install OpenMPI
sudo apt update
sudo apt-get install -y libopenmpi-dev
#sudo apt install -y openmpi-bin # This doesn't work anymore

# Install horovod
#HOROVOD_WITH_TENSORFLOW=1 HOROVOD_GPU_OPERATIONS=NCCL HOROVOD_WITH_MPI=1 python3.8 -m pip install --no-cache-dir horovod[tensorflow,keras]
#export PATH="/home/ubuntu/.local/bin:$PATH" # This is only for the current session
HOROVOD_WITH_TENSORFLOW=1 HOROVOD_GPU_OPERATIONS=NCCL HOROVOD_WITH_MPI=1 pip3 install --no-cache-dir horovod[tensorflow,keras]
export PATH="/home/ubuntu/.local/bin:$PATH"
horovodrun --check-build
horovodrun -np 1 -H localhost:1 python3 keras_mnist.py
