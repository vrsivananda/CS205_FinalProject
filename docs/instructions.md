# Software Instructions

### Download Code

To download the code and the corresponding setup instructions to your local machine, please perform the following:

```
git clone https://github.com/vrsivananda/CS205_FinalProject
```

### Software Setup Instructions

Our pipeline is broken down into the three phases as outlined below. All testing done for these phases is described below, though could, in principle be combined into a single phase. However, each of these steps can be discretized. This is an important aspect when considering cost and the computational resources necessary to sustain our models, as users may wish to divide the resources necessary, either for time purposes or in order to reduce the financial constraints of scraping the data and running the models together.

1. **Data Processing**:  The data processing pipeline retrieves the data from the `yfinance` API, a Python-based framework which accesses minute-by-minute trade data for all 500 stocks of the S&P500 from Yahoo Finance. Using multiple-thread and multiple-core processing, this pipeline is sped up significantly. 

   1. A more detailed description of the design and implementation of the data processing phase can be found **[here](https://github.com/vrsivananda/CS205_FinalProject/blob/master/docs/processing.md)**
   2. The setup instructions for the data processing phase can be found **[here](https://github.com/vrsivananda/CS205_FinalProject/blob/master/stock_predict/data_proc/README.md)**

   

2. **Model Training**: To model this high-frequency sequential data, we create sequences of 60 minutes each, and feed these sequences into a version of a recurrent neural network known as a LSTM, or Long Short-Term Memory. These models, which exploit the sequential nature of the data, are difficult to parallelize. Thus, we implement two strategies: first, we use accelerated computing on a GPU to speed up matrix-multiplication, and second we orchestrate (a) multiple GPUs on a single node, and (b) up to four GPUs across multiple nodes. As discussed in the Model Training section, our code and framework are also scalable to the multi-GPU, multi-node case. 

   1. A more detailed description of the design and implementation of the model training phase can be found **[here](https://github.com/vrsivananda/CS205_FinalProject/blob/master/docs/model_training.md)**
   2. The setup instructions for the model training phase can be found **[here](https://github.com/vrsivananda/CS205_FinalProject/blob/master/stock_predict/models/README.md)**

   

3. **Prediction**: Once the model has been trained, the next logical step is to serve predictions. We do so in a streamed manner. That is, each minute, we take new prices and volume for some number of stocks and serve predictions of the 5-minute ahead price, each minute.

   1. A more detailed description of the design and implementation of the prediction phase can be found **[here](https://github.com/vrsivananda/CS205_FinalProject/blob/master/docs/prediction.md)**
   2. The setup instructions for the prediction phase can be found **[here](https://github.com/vrsivananda/CS205_FinalProject/blob/master/stock_predict/predict/README.md)**

