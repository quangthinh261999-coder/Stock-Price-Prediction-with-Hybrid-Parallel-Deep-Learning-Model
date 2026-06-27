# Hybrid Parallel Deep Learning Model with Attention Mechanisms for Stock Price Prediction

This repository contains the partial implementation accompanying our work on stock price forecasting using a **Hybrid Parallel Deep Learning (HPDL)** model that combines CNN, LSTM/BiLSTM, and attention mechanisms.

**Published paper:** [IEEE Xplore](https://ieeexplore.ieee.org/document/11360207/authors#authors)

## Authors
- **Nguyen Quang Thinh** — Faculty of Engineering Technology, Yersin University, Da Lat City, Vietnam (quangthinh261999@gmail.com)
- **Nguyen Kim Quoc** — Faculty of Information Technology, Nguyen Tat Thanh University, Ho Chi Minh City, Vietnam (nkquoc@ntt.edu.tw)

## Abstract
Deep learning architectures have outperformed conventional methods in time series analysis. This study explores a parallel combination of deep learning models incorporating attention mechanisms for stock price prediction, focusing on Oracle Corporation (ORCL) stock. The proposed model, called the **Hybrid Parallel Deep Learning (HPDL)** model, learns diverse knowledge from parallel component branches combined through an attention-based fusion strategy, achieving superior results compared to single deep learning models. Using Oracle's historical stock price data from 2020 to 2024, the paper compares the predictive performance of various architectures (LSTM, BiLSTM, CNN-LSTM, CNN-BiLSTM, their attention-augmented variants, and parallel attention-based ensembles), analyzing the effects of hyperparameters and optimization techniques. Results demonstrate that hybrid and parallel models accurately capture complex temporal patterns and outperform baselines from prior research.

## Model Overview
The HPDL architecture integrates three core components running in **two parallel processing streams**:
1. **1D CNN** — extracts local temporal patterns from the input sequence
2. **LSTM / BiLSTM** — models long-term sequential dependencies
3. **Attention mechanism** — dynamically weighs the most relevant timesteps

The context vectors from both parallel branches are concatenated and passed through fully connected layers to produce the final prediction.

**Input features:** Open, High, Low, Close (Close as the prediction target)
**Preprocessing:** Min-Max normalization, 90/10 train-test split

## Dataset
- **Source:** Oracle Corporation (ORCL) historical stock data, retrieved from Yahoo Finance
- **Period:** December 31, 2019 – May 29, 2024 (1,100 observations)
- **Features:** Date, Open, High, Low, Close, Adjusted Close, Volume
- **Split:** 90% training / 10% testing (training set further split for validation)

## Key Results
Models were grouped into three categories: **baseline** (LSTM, BiLSTM, CNN-LSTM, CNN-BiLSTM), **attention-augmented** baselines, and **parallel attention-based ensembles**. Evaluation metrics: RMSE, MSE, MAE, MAPE, R².

| Model | Lookback | MAE | MAPE | R² |
|---|---|---|---|---|
| LSTM (baseline) | 7 days | 1.50204 | 1.208 | 0.89756 |
| CNN-BiLSTM | 7 days | 1.40285 | 1.561 | 0.91491 |
| CNN-BiLSTM-Attention | 14 days | 1.34355 | 1.149 | 0.92171 |
| **Parallel CNN-BiLSTM-Attention** | **7 days** | **1.23966** | **1.061** | **0.93962** |

The **Parallel CNN-BiLSTM-Attention** model achieved the best overall performance, confirming that combining hybrid feature extraction, bidirectional temporal modeling, attention, and parallel ensembling yields the most accurate and stable forecasts.

The proposed model was also compared against prior work on Apple (AAPL) stock data, where the best configuration (Parallel CNN-BiLSTM-Attention, 14-day lookback) achieved an MAE of 1.65142 and R² of 0.98261, outperforming the best baseline reported in [Tan et al., ICTCC 2024] (MAE 2.2685, R² 0.9672).

## Citation
If you find this work useful, please cite:
```bibtex
@INPROCEEDINGS{11360207,
  author={Quoc, Nguyen Kim and Thinh, Nguyen Quang},
  booktitle={2025 19th International Conference on Advanced Computing and Analytics (ACOMPA)}, 
  title={Hybrid Parallel Deep Learning Model with Attention Mechanisms for Stock Price Prediction}, 
  year={2025},
  volume={},
  number={},
  pages={29-36},
  keywords={Deep learning;Analytical models;Attention mechanisms;Accuracy;Computational modeling;Time series analysis;Predictive models;Data models;Forecasting;Optimization;Long- and short-term memory;attention mechanisms;and parallel deep learning models},
  doi={10.1109/ACOMPA69200.2025.00012}}
```

## Acknowledgment
We acknowledge Nguyen Tat Thanh University, Ho Chi Minh City, Vietnam, for supporting this study.
