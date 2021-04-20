#!/bin/bash

## Executes processing pipeline
python3 process_prices.py > failed_download_messages

## Move to AWS bucket
aws s3 cp s3://cs205-stream-stock-predict

