## Data Processing
### Overview
Scripts to process data. Note that output data will be put to `../data/` directory. Additionally, due to `yfinance` data processing limits, this process will take only the previous 30 days of data for predictions at the 1 minute interval. Any further model updates would necessarily be batched, beginning in the stream prediction phase. For example, instead of only serving predictions, that mode could be adapted such that the data and outcomes are stored over time and used to update the model periodically. 

#### Instructions

The data processing can be quite extensive and is generally limited by access to `yfinance`. Moreover, while the data is pulled for many stocks, our sequence generation algorithm is embarrassingly parallel; each ticker-day combination can be processed independently. One final note: the data has been stored in an S3 bucket for convenience and collaboration. As noted below, the 

##### Locally

To be compatible across both local and cloud installations, all commands assume a Unix shell, and have been encapsulated in the following two shell scripts: `processing_setup.sh` and `processing_pipeline.sh`. Note that the second script assumes that the AWS CLI has been set up to post to the S3 bucket, but can be commented out to simply process the data.

1. Install Python (Version 3.8.5 or later)
2. Move `requirements.txt` to directory of `processing_setup.sh`
3. Execute the following bash command:

```bash
./processing_setup.sh
```

4. The data processing pipeline runs via a shell script which can take a variable number of tickers to process. To process all tickers, the command line argument should be `all`
   i.  All tickers:

```bash
./processing_pipeline.sh all
```
   	ii. Process N tickers

```bash
./processing_pipeline.sh N
```

##### Cloud (AWS EC2)

###### Instance Details

###### Instructions

1. Connect to EC2 instance
2. Transfer `processing_pipeline.sh`, `process_prices.py`, `requirements.txt`and `processing_setup.sh` files
3. Execute:
```bash
./processing_setup.sh
```
4. Check connection with AWS S3 bucket:
	- Can be verified with `aws s3 ls`
	- If not configured, `aws configure` will allow for input of access and secret access keys, which must be generated in the AWS IAM manager. To generate access and secret access keys, please follow [here](https://docs.aws.amazon.com/cli/latest/userguide/cli-chap-configure.html).
5. As with the local implementation, this can be run for 'all' tickers in the S&P500 or only a subset (N) by running one of the following commands, where `N` is the number of tickers to process:

```bash
./processing_pipeline.sh all
./processing_pipeline.sh N
```
