#!/bin/bash

## Installs necessary files to run data processing
sudo apt update
sudo apt install software-properties-common

# Install Python & AWS CLI
sudo apt-get install python3.8.5
python3.8 --version
sudo apt install python3-pip
sudo apt install awscli

# Install CMake
sudo apt-add-repository universe
sudo apt-get update
sudo apt-get install -y cmake

# Install necessary Python packages
#pip3 install --upgrade pip
pip3 install --no-cache-dir  -r prediction_requirements.txt 

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
echo 'export PATH="$PATH:/usr/local/spark/bin"' >> ~/.profile

source ~/.profile