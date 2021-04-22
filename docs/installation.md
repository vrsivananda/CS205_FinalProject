# Installation Guide

### Data Processing

The data processing module can be run on any AWS instance. Because the main overhead associated with this processing is communication with the `yfinance` API, this process is bound by the number of available processes. The default assumption for processing without re-configuration of the entire pipeline is an AWS EC2 t2.2xlarge instance, with 8 vCPUs and 32GB of memory.

###### Instructions for replication

- Instantiate `t2.2xlarge` instance, using Ubuntu 20.04 on AWS & connect via SSH.
- Upload `processing_setup.sh` and `requirements.txt`. These file will install Java, Scala, Spark, Python, as well as the necessary Python packages to run our data processing.
  - Once finished, the versions of Java, Scala, and Python can be confirmed
  - To enable Spark on the cluster, add the following line to `./profile`:

```bash
export PATH="$PATH:/usr/local/spark/bin"
```
- Next, add the private IP address information to: `/etc/hosts`

