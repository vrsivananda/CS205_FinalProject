## Data Processing
### Overview
Scripts to process data. Note that output data will be put to `../data/` directory. Additionally, due to `yfinance` data processing limits, data is written iteratively and will be stored according to `prevdate` file.



#### Instructions

The data processing can be quite extensive and is generally limited by access to `yfinance`. Moreover, while the data is pulled for many stocks, our sequence generation algorithm is embarrassingly parallel; each ticker-day combination can be processed independently. One final note: the data has been stored in an S3 bucket for convenience, but is of substantial size. Therefore, default `process_prices.py` script processes only a single ticker for debugging purposes.

##### Locally

To be compatible across both local and cloud installations, all commands assume a Unix shell, and have been encapsulated in the following two shell scripts: `setup_environment.sh` and

1. Install Python (Version 3.8.5 or later)
2. 

##### Cloud (AWS EC2)

###### Instance Details

###### Instructions

1. Connect to EC2 instance
2. Transfer `processing_pipeline.sh`, `process_prices_aws.py`, `requirements.txt`and `last_date` files
3. Execute:
```bash
./processing_pipeline.sh
```
4. Check connection with AWS S3 bucket:
	- Can be verified with `aws s3 ls`
	- If not configured, `aws configure` will allow for input of access and secret access keys, which must be generated in the AWS IAM manager. To generate access and secret access keys, please follow [here](https://docs.aws.amazon.com/cli/latest/userguide/cli-chap-configure.html).






