# DeepMeth
Source code for AAAI 2022 paper: ["Noninvasive Lung Cancer Early Detection via Deep Methylation Representation Learning"](#DeepMeth).

Table of Contents
=================
  * [Introduction](#Introduction)
  * [Dataset](#Dataset)
  * [Environments](#Environments)
  * [Dependencies](#Dependencies)
  * [Usage](#Usage)
  * [Citation](#Citation)

## 0. Introduction

## 1. Dataset
The main fields of raw data include 


## 2. Environments
- Python (3.7.7)
- CUDA (10.2)
- Ubuntu-18.04 (5.4.0-91-generic)

## 3. Dependencies
- numpy
- pandas
- torch
- tensorboardX
- tqdm

## 4. Usage
### Data Preprocess
Before start the model training, run the following python script to preprocess the raw datas.

``` Bash
>> python preprocess/tnc_encode.py 
    --pattern_dirs RAW_PATTERN_FILE_DIR1 [--pattern_dirs RAW_PATTERN_FILE_DIR2] ...
    --tnc_dir ENCODE_OUTPUT_DIR
    --n_threads 8
```

### Data Preparation
Then prepare the data for later training. Modify the `prepare` parameters in file [config_example.yml](./config_example.yml). 

Run `run_prepare.sh` in bash.
```Bash
>> sh run_prepare.sh [SPLIT_START_IDX] [SPLIT_END_IDX]
```
### Autoencoder

#### Training
Modify the `auto_encoder_train` parameters in file [config_example.yml](./config_example.yml).

Then run `run_train.sh` in bash.
```Bash
>> sh run_train.sh [SPLIT_START_IDX] [SPLIT_END_IDX]
```

#### Encode
Modify the `auto_encoder_encode` parameters in file [config_example.yml](./config_example.yml).

Then run `run_encode.sh` in bash.
```Bash
>> sh run_encode.sh [SPLIT_START_IDX] [SPLIT_END_IDX]
```

You can also merge the Data Preparation, Autoencoder training and encoding bash scripts in to one script.

## 5. Citation
<!-- If you use this work or code, please kindly cite the following paper: -->
