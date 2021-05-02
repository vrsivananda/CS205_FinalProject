#!/bin/bash

## Executes processing pipeline
python3 process_prices.py $1 $2

## Move to AWS bucket
aws s3 cp training_data.npz s3://cs205-stream-stock-predict

