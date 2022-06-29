
# DivideFlex
A simple method for 1st Learning and Mining with Noisy Labels Challenge  IJCAI-ECAI 2022

## Dependencies
You can use requirements.txt for building dependencies.

## Usage

1. You can run run.sh with 5 pre-fixed seeds. Each run will be evaluated w.r.t. a random selected subset of CIFAR-10 test data with replacement, and take the average performance of 5 runs.
2. run learning.py
```
python learning.py --load_path last_model.pth --noise_type [noise_type]
```
