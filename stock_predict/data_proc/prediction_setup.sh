#!/bin/bash

## Installs necessary files to run data processing
sudo apt update
sudo apt install software-properties-common

# Install Python & AWS CLI
sudo add-apt-repository -y ppa:deadsnakes/ppa
sudo apt install -y python3.8
python3.8 --version
sudo apt install -y python3-pip
sudo apt install awscli

# Install CMake
sudo apt-add-repository universe
sudo apt-get update
sudo apt-get install -y cmake

# Install necessary Python packages
python3.8 -m pip install --upgrade pip
python3.8 -m pip install -r prediction_requirements.txt --no-cache-dir 

# Install Spark precursors
sudo apt-add-repository ppa:webupd8team/java
sudo apt-get update
sudo apt install openjdk-8-jdk
sudo apt-get install scala

# Install Spark
sudo curl -O https://downloads.apache.org/spark/spark-3.1.1/spark-3.1.1-bin-hadoop2.7.tgz
sudo tar xvf ./spark-3.1.1-bin-hadoop2.7.tgz
sudo mkdir /usr/local/spark
sudo cp -r spark-3.1.1-bin-hadoop2.7/* /usr/local/spark
export 'PATH="$PATH:/usr/local/spark/bin"' >> ~/.profile
source ~/.profile