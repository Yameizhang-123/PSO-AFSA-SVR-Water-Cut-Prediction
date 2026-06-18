# PSO-AFSA-SVR-Water-Cut-Prediction
# PSO-AFSA-SVR-Water-Cut-Prediction

## Overview
This repository contains the source code for the hybrid optimized support vector regression model proposed in the paper *Water Cut Prediction with Hybrid Optimized SVR*. It implements a two-stage hyperparameter tuning strategy combining particle swarm optimization (PSO) and artificial fish swarm algorithm (AFSA), with sliding time window support for oilfield water cut forecasting.

## Requirements
- Python 3.7+
- scikit-learn ≥ 1.0
- numpy ≥ 1.21
- pandas ≥ 1.3

## Usage
1. Prepare the production dynamic dataset with 6 input features as described in the paper.
2. Run the main script to train the model and output prediction metrics (MAPE, MAE, RMSE).
3. Sliding window size and algorithm parameters can be adjusted in the configuration section.

## License
MIT License
