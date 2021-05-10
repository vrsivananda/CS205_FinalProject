# Phase I: Data Processing

#### Overview

The data processing for this problem represents a case of high-throughput computing. The major challenge of this section is communication with the `yfinance` API. Because this API provides data in regimented ways, the processing described below is specifically suited to its design. In general, this is a feature of data processing when working with external sources, but provides a platform to creatively speed up the process. Additionally, one note about `yfinance` is that only 30 days of minute-by-minute data are stored. Moving beyond the proof-of-concept of this project, we envision the most efficient scraping tool for this project to collect approximately once a day, for only one days worth of data. In that case, though, the speedup here may be less relevant. Additionally, `yfinance` has most tickers available on Yahoo Finance, but is inherently limited. We have selected the stocks of the S&P500 as our baseline, though this could surely be expanded, and we would expect similar speedup.

#### Data Sources

In this project, we primarily rely on data from `yfinance`, a Python library which interacts with Yahoo Finance-provided market information at many resolutions. Specifically, our training data consists of the the Close Price and Trade Volume on a minute-by-minute basis, for 503 of the 505 stocks in the S&P 500, which includes most major US-based corporations. The financial data covers the trading days occurring between Monday, April 4, 2021 and Friday, April 30, 2021. One important clarification is that `yfinance` makes only 30 days of data available (this corresponds to fewer trading days). 

#### Programming Model & Parallelism

Much of the data processing centers around the `yfinance` API (see [here](https://pypi.org/project/yfinance/)). This is an API written in Python to access stock price and volume information down to the minute level. Because this application is written in Python, the scraper to communicate with the API must be in Python (or through Python bindings in another language). Given the global interpreter lock (GIL) and Python's limits on multicore and multithread computation, this was necessary a tricky place to implement parallelism. However, due to the problem's high-throughput and nearly embarrassingly parallel structure, we were able to make use of modules such as `multiprocessing` and `multitasking`, though the latter is implicitly built into the `yfinance` API call. Thus, we effectively discretized our stocks of interest into separate batches. Each batch was spawned into its own process, using multithreading to access the information from many stocks at once. From there, the data are effectively split into stock-day components, and sequences are generated and concatenated together to form the training dataset. All testing and processing was implemented on an AWS `t2.2xlarge` instance, with 8 vCPU and 32 GB of memory. Please see the end of this discussion for extended replicability details.

##### Problem description

Most sources of financial market information store those data on the basis of a unique identifier. In the case of stocks, this is a unique ticker, though our methods could be easily extended to the much larger bond market, where CUSIPs serve in a similar capacity. In our project, we use high-frequency data, at the resolution of one minute. Due to the limits of the API design, only 30 days of data are available. We note that our method is generalizable, though, and if more data were available, would be able to be extended without only minor adaptations. Thus, we seek to download data for each stock in the S&P500, on each trading day within the previous 30 days, then convert these data to 60 minute sequences to predict 5 minutes after the end of the sequence. 



Note that this problem is nearly embarrassingly parallel: because we do not overlap days (i.e. we do not concatenate a sequence of the final 30 minutes of one trading day to the first 30 minutes of the next, due to possibilities for information flows not captured in the model), each stock-day pair is independent. Thus, the general complexity of our processing task is on the order $O(S * D * N)$ where $S$ is the number of unique stock tickers considered, $D$ is the number of days, and $N$ corresponds to the number of sequences in a day. Psuedocode for the naive approach is as follows:

```py
for t in tickers:
	for d in days:
		for n in sequences_per_day:
			seq = generate_sequence(t, d, n)
```

In ac

##### Overheads & Mitigation Strategies

There are a number of overheads for this phase of the project, most notably associated with the API communication. 

1. *Communication with `yfinance` API*: After implementing a simplified, naïve sequential version of our application, code profiling revealed that the `yfinance` `download()` method from the `mulit.py` module and associated `sleep()` command took the bulk of the time. The table below demonstrates the timing results for the naïve approach, for varying tickers as a demonstration. Table 1 reports the total cumulative time, Table 2 reports the per-call time based on the cumulative time, as the method is recursive. Given the obvious need for multithreading, we 

| Number of Tickers | Total Time | Download - Total | Download - Per Call | Sleep - Total | Sleep - Per Call |
| ----------------- | ---------- | ---------------- | ------------------- | ------------- | ---------------- |
| 1                 | 18.956     | 18.009           | 0.621               | 17.950        | 0.01             |
| 5                 | 88.373     | 87.439           | 0.603               | 87.141        | 0.01             |
| 10                | 174.934    | 173.998          | 0.600               | 173.298       | 0.01             |



2. *Date limits*: For 1-minute resolution, `yfinance` stores only 30 days of data total, and only 7 days of data may be accessed in a single call.

3. *Rate limits*: While not clearly documented on the website, our performance results and testing suggest that there are substantial rate limits on the API. To ensure comparability, we ran test runs before our performance comparison to ensure the rate limit had been met, thus ensuring our pulls were 'slower'. As a technical note, our testing suggests that the rate limit would have been hit before finishing one round of our data pull and thus made sense to approach in this manner.

To mitigate these overheads, we implemented the following strategies:

1. *Multiple threading*: The `yfinance` download functionality supports downloads of many Tickers at once through `threading`, which implements many threads at once. This is particularly helpful for I/O bound tasks. Because we are pulling significant data, these tasks are inherently I/O bound. While we could theoretically re-create this behavior, the built-in functionality performs well, as demonstrated in the table below, which is a direct comparison to Table 1. See the end of this instructions for the appropriate files and commands to replicate.

| Number of Tickers | Total Time | Download - Total | Download - Per Call | Sleep - Total | Sleep - Per Call |
| ----------------- | ---------- | ---------------- | ------------------- | ------------- | ---------------- |
| 1                 | 2.596      | 3.050            | 0.058               | 2.996         | 0.011            |
| 5                 | 11.826     | 10.889           | 0.375               | 9.770         | 0.011            |
| 10                | 15.408     | 14.481           | 0.499               | 12.145        | 0.012            |



2. *Multiple-core processing*: While the GIL in Python prevents true multi-core, shared memory processing, this task does not need to be shared memory. In fact, by mapping several independent processes each operating on separate chunks of tickers, we achieve substantial speedup. This is implemented via Python's `multiprocessing` module.

#### Performance evaluation

To evaluate the performance and speedup of these parallel implementations, we conducted three tests, which test the parallelism and performance gains from both multithreading and multiprocessing. The first test compares the relative speedup to the fully naïve baseline. As shown above, this is an inefficient process due to the number of non-parallelized API calls whereas a more parallel approach would use multithreading. Multithreading is implemented as a very fine-grained level of parallelism. Specifically, this is tuned to address I/O issues. The process of retrieving the data for separate tickers can be parallelized, even if ultimately writing that to memory cannot be (i.e. thread-level parallelism). The figure below shows the speedup gained from using many processors, but no multithreading.

![multiprocessing_nothreading](https://github.com/vrsivananda/CS205_FinalProject/blob/master/docs/figures/speedup_singlethread.png)

This test is conducted for only 50 tickers due to the extreme inefficiency. Moreover, because `yfinance` natively supports multithreading, we view the parallelism applied as primarily comparing to a baseline where multithreading is used, but not multiprocessing. We do believe that this is relevant, however, because it represents our first attempt at parallelism -- carefully considering I/O issues is an important part of any data intensive tasks, and parallelizing computation may be completely obviated by additional overheads introduced via I/O. Our first observation is that performance actually degrades for using multiprocessing relative to a baseline without it, highlighting the overhead associated with managing the multiple (albeit single) process. Second, as the number of processes increases, the relative speedup decreases. Ultimately, this is due to the I/O intensive nature of the process, both in terms of retrieving the data, and in the inherently sequential outputs of the data. Importantly, while we considered a distributed data platform such as Spark, we found that the overheads associated with parallelizing the data, as well as the memory overhead for orchestration of a DataFrame, made the process significantly slower. While we have many samples, the total of our data is only approximately 2GB, which can be handled by all but the most lightweight computing platforms.



The second test we undertook is our main parallelization evaluation. This utilized multiple processes to perform the computation while also utilizing multi-threading. The plot below compares performance overall 500 stock tickers, representing a more realistic view of our data processing task. The plot below shows a speedup of approximately 6.85x for the 8 core processing task. The scaling here closely follows Ahmdal's law, which is intuitive given that it is almost embarrassingly parallel. The small reduction in time is likely due to the I/O restrictions of creating the dataset. After processing, the sequences are concatenated into a single dataset to be fed to the model. There are two further points to highlight. First, the multi-process mode with only a single process is still slightly slower (speedup of 0.96x) compared to the sequential version. This slight penalty shows the computational cost associated with orchestration of the `multiprocessing` library and likely the `fork` `join` model of waiting for the process to complete. Finally, the non-parallel version of this process takes approximately 36 minutes compared to slightly more than 5 minutes for the parallel version. Indeed, estimating this performance over the fully sequential version suggests the code would take approximately 2.5 hours! Thus, our solution here dramatically reduces the time needed to pull this data.

![multiprocessing](https://github.com/vrsivananda/CS205_FinalProject/blob/master/docs/figures/full_multithread_perf.png)

Finally, we put our analyses together to compare the full performance gains. While we have computed speedup, we have not added a theoretical component. This is because it is difficult to directly calculate the speedup. We tested our code on an 8 vCPU AWS instance (`t2.2xlarge`). The `yfinance` application of multithreading automatically allocates 2 threads for each CPU, or 16 in total. Because of the I/O-bound nature of this problem, it is not clear that simply adding more threads would present a strong-scaling opportunity. However, over the purely naïve approach, the multithreading and multiprocessing achieves a speedup of nearly 25x.

![all_speedup](https://github.com/vrsivananda/CS205_FinalProject/blob/master/docs/figures/speedup_all.png)

#### Lessons & Future Direction

1. Interaction with external APIs or data sources may set parameters for parallelism outside of the theoretical domain. In practical applications of parallelism, an API may dictate a particular I/O structure. Designing parallel applications can speed these up, but may have to be designed with that specific constraint in mind to achieve the strong scaling performance associated with Ahmdal's law. Moreover, the garden of forking paths is vast: there are many ways to develop parallel programs within these constraints, and can be difficult to evaluate each.
2. Parallel applications can clear up *developer* overhead in addition to solving previously not-solvable issues. Cutting a pipeline from the better part of an hour to approximately five minutes allows for many more questions to be asked and answered.

#### Technical details

##### Replication

All testing was conducted on AWS. Specification details are below:

- Instance: `t2.2xlarge`

| Hardware Spec. | Details                                   |
| -------------- | ----------------------------------------- |
| Model          | Intel(R) Xeon(R) CPU E5-2676 v3 @ 2.40GHz |
| # of vCPU      | 8                                         |
| Cores per CPU  | 8                                         |
| L1 Cache       | 256KB                                     |
| L2 Cache       | 2MB                                       |
| L3 Cache       | 30MB                                      |
| Main Memory    | 32GB                                      |

| Software Spec.   | Details      |
| ---------------- | ------------ |
| Operating System | Ubuntu 20.04 |
| Compiler         | GCC 9.3.0    |
| Python           | 3.8.5        |
| Spark            | 3.1.1        |

| Python Package                                  | Version |
| ----------------------------------------------- | ------- |
| `NumPy`                                         | 1.19.2  |
| `yfinance`                                      | 0.1.59  |
| `pandas`                                        | 1.2.4   |
| `re`                                            | 2.2.1   |
| Other libraries part of Python Standard Library | 3.8.5   |

- Code profiler: cProfile, tables created via cProfile & `grep`

- The profiling code can be found [here](https://github.com/vrsivananda/CS205_FinalProject/tree/master/stock_predict/performance_testing/data_proc/profile_downloads.py). The command to run the profiler for `N` tickers in parallel mode or sequential mode is:

  ```bash
  python3 -m cProfile -s 'tottime' profile_downloads.py N 'parallel' | grep '[ds][ol][we][ne]'
  python3 -m cProfile -s 'tottime' profile_downloads.py N 'sequential' | grep '[ds][ol][we][ne]'
  ```

   

##### Sources

- https://docs.python.org/3/library/threading.html# 