Instructions

1. Spin up t2.2xlarge AWS instance running Ubuntu 20.04 Linux distro

2. Upload `stock_client.py`, `stock_processing.py`, `prediction_setup.sh`, `prediction_requirements.txt`

3. Add internal IP addresses to `/etc/hosts` , using the command `sudo vim /etc/hosts`

4. Execute:

   ````bas
   ./prediction_setup.sh
   ````

5. Update the `~/.profile` to change the paths, using `vim ~/.profile`, according to the instructions below: 

Add the following paths:
`export PATH="/usr/sbin:/usr/bin:/sbin:/bin"`
`export PATH="$PATH:/usr/local/spark/bin"` 

Note: may need to remove the PATH for `/home/ubuntu/.local/bin`

After editing the `~/.profile` file, then execute `source ~/.profile`

6. Test setup with `run-example SparkPi 10`

7. In the AWS Security group, add an inbound rule for All TCP Connections from `0.0.0.0/0`

8. Add `trained_lstm_mod.h5` from either local drive or AWS S3 bucket

   1. If using AWS S3 bucket, AWS credentials will need to be configured. This can be accomplished by executing `aws configure`. The subsequent command to transfer the data is: `aws s3 cp s3://cs205-stream-stock-predict/trained_lstm_mod.h5 .`

9. Open a second terminal & connect via SSH

10. In the first terminal, execute:

```
spark-submit stock_processing.py
```

9. In the second terminal, execute:

```bash
python3 stock_client.py
```

