# Hybrid Parallel Deep Learning Model with Attention Mechanisms for Stock Price Prediction

This repository contains the partial implementation accompanying our work on stock price forecasting using a **Hybrid Parallel Deep Learning (HPDL)** model that combines CNN, LSTM/BiLSTM, and attention mechanisms.

**Published paper:** [IEEE Xplore](https://ieeexplore.ieee.org/document/11360207/authors#authors)

## Authors
- **Nguyen Quang Thinh** — Faculty of Engineering Technology, Yersin University, Da Lat City, Vietnam (quangthinh261999@gmail.com)
- **Nguyen Kim Quoc** — Faculty of Information Technology, Nguyen Tat Thanh University, Ho Chi Minh City, Vietnam

## Abstract
Deep learning architectures have outperformed conventional methods in time series analysis. This study explores a parallel combination of deep learning models incorporating attention mechanisms for stock price prediction, focusing on Oracle Corporation (ORCL) stock. The proposed model, called the **Hybrid Parallel Deep Learning (HPDL)** model, learns diverse knowledge from parallel component branches combined through an attention-based fusion strategy, achieving superior results compared to single deep learning models. Using Oracle's historical stock price data from 2020 to 2024, the paper compares the predictive performance of various architectures (LSTM, BiLSTM, CNN-LSTM, CNN-BiLSTM, their attention-augmented variants, and parallel attention-based ensembles), analyzing the effects of hyperparameters and optimization techniques. Results demonstrate that hybrid and parallel models accurately capture complex temporal patterns and outperform baselines from prior research.

## Model Overview
The HPDL architecture integrates three core components running in **two parallel processing streams**:
1. **1D CNN** — extracts local temporal patterns from the input sequence
2. **LSTM / BiLSTM** — models long-term sequential dependencies
3. **Attention mechanism** — dynamically weighs the most relevant timesteps
<img width="8000" height="4500" alt="PARALLEL CNN BI LSTM" src="https://github.com/user-attachments/assets/ee7a5963-152d-46ed-a6c9-6c19647b7ca7" />


The context vectors from both parallel branches are concatenated and passed through fully connected layers to produce the final prediction.

**Input features:** Open, High, Low, Close (Close as the prediction target)
**Preprocessing:** Min-Max normalization, 90/10 train-test split

## Dataset
- **Source:** Oracle Corporation (ORCL) historical stock data, retrieved from Yahoo Finance
- **Period:** December 31, 2019 – May 29, 2024 (1,100 observations)
- **Features:** Date, Open, High, Low, Close, Adjusted Close, Volume
- **Split:** 90% training / 10% testing (training set further split for validation)
<img width="1489" height="790" alt="oracle_closing_prices" src="https://github.com/user-attachments/assets/133748bf-7e6e-4f88-b59a-79d6456126ae" />


## Key Results
Models were grouped into three categories: **baseline** (LSTM, BiLSTM, CNN-LSTM, CNN-BiLSTM), **attention-augmented** baselines, and **parallel attention-based ensembles**. Evaluation metrics: RMSE, MSE, MAE, MAPE, R².

| Model | Lookback | MAE | MAPE | R² |
|---|---|---|---|---|
| LSTM (baseline) | 7 days | 1.50204 | 1.208 | 0.89756 |
| CNN-BiLSTM | 7 days | 1.40285 | 1.561 | 0.91491 |
| CNN-BiLSTM-Attention | 14 days | 1.34355 | 1.149 | 0.92171 |
| **Parallel CNN-BiLSTM-Attention** | **7 days** | **1.23966** | **1.061** | **0.93962** |
<img width="6342" height="4305" alt="PARALLEL MODELS ATTENTION CHART" src="https://github.com/user-attachments/assets/92d7bd28-b318-4af0-b841-bec290be8041" />


The **Parallel CNN-BiLSTM-Attention** model achieved the best overall performance, confirming that combining hybrid feature extraction, bidirectional temporal modeling, attention, and parallel ensembling yields the most accurate and stable forecasts.

The proposed model was also compared against prior work on Apple (AAPL) stock data, where the best configuration (Parallel CNN-BiLSTM-Attention, 14-day lookback) achieved an MAE of 1.65142 and R² of 0.98261, outperforming the best baseline reported in [Tan et al., ICTCC 2024] (MAE 2.2685, R² 0.9672).

## Comparison with Related Work
To validate the objectivity and effectiveness of the proposed models, this study **kept the same model architectures and training configurations** developed for the Oracle (ORCL) dataset, then re-applied them to the **datasets used in related papers**, comparing the results using standard metrics (RMSE, MAE, MAPE, R²).
### 1. Comparison with *"Comparing LSTM Models for Stock Market Prediction: A Case Study with Apple's Historical Prices"* (Hà Minh Tân et al., 2024)
- Dataset: **Apple (AAPL)** stock, 2019–2023.
  <img width="1488" height="790" alt="apple_closing_prices" src="https://github.com/user-attachments/assets/f903fd09-4bfd-4377-bf24-337ba7293e90" />

- Original result (LSTM, 21-day lookback): MAE = 2.2685, R² = 0.9672.
- Re-implemented result: LSTM (14-day lookback) achieved MAE = 1.95362, R² = 0.98031.
- Best model in this study: **Parallel-CNN-Bi LSTM-Attention (14-day lookback)** achieved MAE = 1.65142, R² = 0.98261 — outperforming the original model.
  

### 2. Comparison with *"Stock Price Prediction Using CNN-BiLSTM-Attention Model"* (Zhang, Ye & Lai, 2023 – *Mathematics* journal)
- Dataset: **CSI 300 Index** (China), 2011–2021.
<img width="1486" height="790" alt="csi300_closing_prices" src="https://github.com/user-attachments/assets/ed9fb1ed-7a24-427f-88f2-e308dad0a717" />

| Model | Original Paper (MAPE / RMSE / R²) | Re-implemented (MAPE / RMSE / R²) |
|---|---|---|
| LSTM | 1.877% / 108.748 / 0.958 | 1.320% / 81.759 / 0.976 |
| CNN-LSTM | 1.482% / 89.048 / 0.972 | 1.125% / 66.872 / 0.984 |
| CNN-LSTM-Attention | 1.288% / 76.454 / 0.979 | 1.028% / 64.628 / 0.985 |
| CNN-Bi LSTM-Attention | 1.023% / 64.848 / 0.985 | 0.993% / 62.626 / 0.986 |

<img width="8000" height="4500" alt="CSI300 PARALLEL CNN-BI LSTM-ATTENTION" src="https://github.com/user-attachments/assets/f6a3a0cb-3e8b-4802-8fa0-1f8daebbdbf4" />


For all four shared models, the re-implemented results in this study **outperformed** those reported in the original paper.

### Overall Conclusion
- Adding an **Attention mechanism**, and especially using a **parallel architecture**, significantly improves prediction accuracy compared to single or sequential hybrid models.
- The proposed models generalize well **beyond the Oracle (ORCL) dataset**, also performing strongly on AAPL and CSI 300 data, suggesting broad applicability for financial time-series forecasting.

Note: These comparisons are based on re-implementing the architectures as described in the original papers. Results may differ slightly due to differences in environment, library versions, or data preprocessing details not fully disclosed in the original publications.

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
