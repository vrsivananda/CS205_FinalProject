#!/bin/bash

## Executes processing pipeline
python3 process_prices.py $1 > failed_download_messages
tail -n 2 failed_download_messages > processing_timing
cat processing_timing
rm failed_download_messages

## Move to AWS bucket
aws s3 cp training_data.npz s3://cs205-stream-stock-predict

