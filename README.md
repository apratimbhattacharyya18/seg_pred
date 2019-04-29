# Introduction
Code for the paper: [Bayesian Prediction of Future Street Scenes using Synthetic Likelihoods](https://openreview.net/forum?id=rkgK3oC5Fm) 

# Requirements

* Python2.7
* h5py
* Tensorflow 1.1.0
* Keras 2.0.3
* tqdm

# Data
* Data required for training can be obtained by running the `download_data.sh` script.

# Training
* Run `main.py`. All data must be present in the data directory. Set `test_examples` in `main.py` to control the number of examples from the validation set the model is tested. Set `test_samples` to control the number of samples per test example.