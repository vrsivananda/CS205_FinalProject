#!/bin/bash

## Installs necessary files to run data processing
sudo apt update
sudo apt install software-properties-common
sudo apt-get install python3.8.5
sudo apt install python3-pip

# Install necessary Python packages
pip3 install -r requirements.txt

