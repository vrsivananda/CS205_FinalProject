#!/bin/bash

## Installs necessary files to run data processing
sudo apt update
sudo apt install software-properties-common
sudo apt-get install python3.8.5
sudo apt install python3-pip
sudo apt install awscli
sudo apt-get install cmake

# Install necessary Python packages
pip3 --no-cache-dir install -r processing_requirements.txt

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