# Phase I: Data Processing

#### Overview

The data processing for this problem represents a case of high-throughput computing. The major challenge of this section is communication with the `yfinance` API. Because this API provides data in regimented ways, the processing described below is specifically suited to its design. In general, this is a feature of data processing when working with external sources, but provides a platform to creatively speed up the process. Additionally, one note about `yfinance` is that only 30 days of minute-by-minute data are stored. Moving beyond the proof-of-concept of this project, we envision the most efficient scraping tool for this project to collect approximately once a day, for only one days worth of data. In that case, though, the speedup here may be less relevant. Additionally, `yfinance` has most tickers available on Yahoo Finance, but is inherently limited. We have selected the stocks of the S&P500 as our baseline, though this could surely be expanded, and we would expect similar speedup.

#### Programming Model & Parallelism

Much of the data processing centers around the `yfinance` API (see [here](https://pypi.org/project/yfinance/)). This is an API written in Python to access stock price and volume information down to the minute level. Because this application is written in Python, the scraper to communicate with the API must be in Python (or through Python bindings in another language). Given the global interpreter lock (GIL) and Python's limits on multicore and multithread computation, this was necessary a tricky place to implement parallelism. However, due to the problem's high-throughput and nearly embarrassingly parallel structure, we were able to make use of modules such as `multiprocessing` and `multitasking`, though the latter is implicitly built into the `yfinance` API call. Thus, we effectively discretized our stocks of interest into separate batches. Each batch was spawned into its own process, using multithreading to access the information from many stocks at once. From there, the data are effectively split into stock-day components, and sequences are generated and concatenated together to form the training dataset. All testing and processing was implemented on an AWS `t2.2xlarge` instance, with 8 vCPU and 32 GB of memory. Please see the end of this discussion for extended replicability details.

##### Problem description

Most sources of financial market information store those data on the basis of a unique identifier. In the case of stocks, this is a unique ticker, though our methods could be easily extended to the much larger bond market, where CUSIPs serve in a similar capacity. In our project, we use high-frequency data, at the resolution of one minute. Due to the limits of the API design, only 30 days of data are available. We note that our method is generalizable, though, and if more data were available, would be able to be extended without only minor adaptations. Thus, we seek to download data for each stock in the S&P500, on each trading day within the previous 30 days, then convert these data to 60 minute sequences to predict 5 minutes after the end of the sequence. 



Note that this problem is nearly embarrassingly parallel: because we do not overlap days (i.e. we do not concatenate a sequence of the final 30 minutes of one trading day to the first 30 minutes of the next, due to possibilities for information flows not captured in the model), each stock-day pair is independent. Thus, the general complexity of our processing task is on the order $O(S * D * N)$ where $S$ is the number of unique stock tickers considered, $D$ is the number of days, and $N$ corresponds to the number of sequences in a day.

##### Overheads & Mitigation Strategies

There are a number of overheads for this phase of the project, most notably associated with the API communication. 

1. *Communication with `yfinance` API*: After implementing a simplified, naive sequential version of our application, code profiling revealed that the `yfinance` `download()` and associated `sleep()` command took the bulk of the time. The table below demonstrates the timing results for the naive approach, for varying tickers as a demonstration:

| Number of Tickers | Download Time | Sleep Time |
| ----------------- | ------------- | ---------- |
|                   |               |            |
|                   |               |            |
|                   |               |            |

2. *Date limits*: For 1-minute resolution, `yfinance` stores only 30 days of data total, and only 7 days of data may be accessed in a single call.
3. *Rate limits*: While not clearly documented on the website, our performance results and testing suggest that there are substantial rate limits on the API. To ensure comparability, we ran test runs before our performance comparison to ensure the rate limit had been met, thus ensuring our pulls were 'slower'. As a technical note, our testing suggests that the rate limit would have been hit before finishing one round of our data pull and thus made sense to approach in this manner.

To mitigate these overheads, we implemented the following strategies:

1. *Multiple threading*: The `yfinance` download functionality supports downloads of many Tickers at once through `threading`, which implements many threads at once. This is particularly helpful for I/O bound tasks. Because we are pulling significant data, these tasks are inherently I/O bound. While we could theoretically re-create this behavior, the built-in functionality performs well, as demonstrated in the table below:

## TODO

2. *Multiple-core processing*: While the GIL in Python prevents true multi-core, shared memory processing, this task does not need to be shared memory. In fact, by mapping several independent processes each operating on separate chunks of tickers, we achieve substantial speedup. This is implemented via Python's `multiprocessing` module.

#### Performance evaluation

## TODO

#### Lessons & Future Direction

1. Interaction with external APIs or data sources may set parameters for parallelism outside of the theoretical domain.
2. The garden of forking paths is strong: there are many ways to develop parallel programs within these constraints, and can be difficult to evaluate each. Important to evaluate resources in conjunction with program design.

#### Technical details

##### Replication

All testing was conducted on AWS. Specification details are below:

- 

- Code profiler: cProfile, tables created via cProfile & `grep`

##### Sources

- https://docs.python.org/3/library/threading.html# 