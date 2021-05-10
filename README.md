# Parallelizing LSTM training and online model predictions

## CS205, Spring 2021 - Group 11

- Kevin Hare
- Junkai Ong
- Sivananda Rajananda

## Overview 

In this project, we implement parallelization across the pipeline for data processing, model training, and serving model predictions. This utilizes techniques from across the big data and big compute paradigms -- using multiple core computing to speed up data processing, accelerated computing and distributed memory techniques to improve the training time of an LSTM model while maintaining forecast accuracy, and finally develop a streaming application capable of efficiently serving model predictions of the five-minute ahead price of each stock in the S&P500. Moreover, these methods serve as a proof-of-concept for an even larger model computation. As discussed in our conclusion, these techniques and applications are scalable for even larger models and more fine-grained model timing.

## Logistics

A project table of contents can be found below. Additionally, all software for this project can be found in the `stock_predict` directory of this repository. While we have aggregated instructions for running our software [here](https://github.com/vrsivananda/CS205_FinalProject/blob/master/docs/instructions.md), each subdirectory of the software section contains directory-specific information for running our software. Finally, we plan to maintain our data and trained model used for prediction in the group S3 bucket. We are happy to share this for full-transparency, as our data collection methodology will pull the most up-to-date data.

## TOC

1. [Problem Statement & Background](https://github.com/vrsivananda/CS205_FinalProject/blob/master/docs/motivation.md)
3. [Phase I: Data Processing](https://github.com/vrsivananda/CS205_FinalProject/blob/master/docs/processing.md)
4. [Phase II: Model Training](https://github.com/vrsivananda/CS205_FinalProject/blob/master/docs/model_training.md)
5. [Phase III: Online Prediction](https://github.com/vrsivananda/CS205_FinalProject/blob/master/docs/prediction.md)
6. [Discussion](https://github.com/vrsivananda/CS205_FinalProject/blob/master/docs/discussion.md)
7. Reproducibility Info
8. [Software Instructions](https://github.com/vrsivananda/CS205_FinalProject/blob/master/docs/instructions.md)
9. [Sources](https://github.com/vrsivananda/CS205_FinalProject/blob/master/docs/aggregated_sources.md)
10. [Project Presentations](https://github.com/vrsivananda/CS205_FinalProject/blob/master/presentations/)
