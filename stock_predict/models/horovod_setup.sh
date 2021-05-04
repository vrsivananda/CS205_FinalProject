
# Run this in the terminal to remove the spurious CR characters (added from Windows machine)
#sed -i -e 's/\r$//' horovod_setup.sh

# Increase volume
#sudo apt install -y awscli
#aws ec2 modify-volume --size 128 --volume-id i-065e7c9e4d78a3ee5
# ^ Unused because need to configure aws and manually insert volume id every time we run it
df -h
lsblk
sudo growpart /dev/xvda 1
sudo resize2fs /dev/xvda1

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

# Install pip
sudo apt install -y python3-pip
python3.8 -m pip install --upgrade pip
# ^ Stuff on pip needs to be installed like this instead for Python3.8 to find them

# Install tensorflow
python3.8 -m pip install --upgrade tensorflow
python3.8 -c "import tensorflow as tf;print(tf.reduce_sum(tf.random.normal([1000, 1000])))"

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

# Install NCCL2
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/cuda-ubuntu1804.pin
sudo mv cuda-ubuntu1804.pin /etc/apt/preferences.d/cuda-repository-pin-600
sudo apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/7fa2af80.pub
sudo add-apt-repository "deb https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/ /"
sudo apt-get update
sudo apt install libnccl2=2.9.6-1+cuda11.3 libnccl-dev=2.9.6-1+cuda11.3

# Install OpenMPI
sudo apt update
sudo apt-get install -y libopenmpi-dev
#sudo apt install -y openmpi-bin # This doesn't work anymore

# Install horovod
HOROVOD_WITH_TENSORFLOW=1 HOROVOD_GPU_OPERATIONS=NCCL HOROVOD_WITH_MPI=1 python3.8 -m pip install --no-cache-dir horovod[tensorflow,keras]
export PATH="/home/ubuntu/.local/bin:$PATH" # This is only for the current session
horovodrun --check-build
horovodrun --gloo -np 1 -H localhost:1 python3.8 keras_mnist.py
# ^ This runs with gloo

# Run with Open MPI on one node
mpirun -np 1 -bind-to none -map-by slot -x NCCL_DEBUG=INFO -x LD_LIBRARY_PATH -x PATH -mca pml ob1 -mca btl ^openib python3.8 keras_mnist.py
# ^ This runs with OpenMPI
#mpirun -np 1 -bind-to none -map-by slot -x NCCL_DEBUG=INFO -x PATH -mca pml ob1 -mca btl ^openib python3.8 keras_mnist.py

# Run with Open MPI on multiple nodes
mpirun -np 2 -H master:1,node1:1 -bind-to none -map-by slot -x NCCL_DEBUG=INFO -x LD_LIBRARY_PATH -x PATH -mca pml ob1 -mca btl ^openib python3.8 keras_mnist.py
#mpirun -np 2 -H master:1,node1:1 -bind-to none -map-by slot -x NCCL_DEBUG=INFO -x PATH -mca pml ob1 -mca btl ^openib python3.8 keras_mnist.py


# ---------- Not used ----------

# Removed this because it caused errors
#HOROVOD_WITH_MPI=1

# cmake
#sudo apt remove --purge --auto-remove -y cmake
#wget -O - https://apt.kitware.com/keys/kitware-archive-latest.asc 2>/dev/null | gpg --dearmor - | sudo tee /etc/apt/trusted.gpg.d/kitware.gpg >/dev/null
#sudo apt update
#sudo apt install kitware-archive-keyring
#sudo rm /etc/apt/trusted.gpg.d/kitware.gpg
#sudo apt update
#sudo apt install -y cmake

# Install MPICH
#sudo apt-get install -y libcr-dev mpich mpich-doc
#mpiexec --version

#pip install --upgrade tensorflow
#sudo apt install python-pip
#sudo apt-get install python3.8.5

# Create virtual environment
#sudo apt install python3-dev python3-pip python3-venv
#sudo python3 -m venv --system-site-packages ./venv
#source ./venv/bin/activate  # sh, bash, or zsh
#pip install --upgrade pip #Not installed
#pip list  # show packages installed within the virtual environment

# To deactivate environment
#deactivate  # don't exit until you're done using TensorFlow

# Install necessary Python packages
#pip3 --no-cache-dir install -r requirements.txt
