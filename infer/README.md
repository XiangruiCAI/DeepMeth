# Inference

单次实验的目录结构如下图所示。![目录格式](https://tva1.sinaimg.cn/large/006tNbRwly1g9jm2w0n77j316z0ijtam.jpg)

# 可用的函数

##  tnc_encode

会将pattern文件中的TNC编码成123，存放在`tnc_dir`下

> command line: 
>
> `python -m infer tnc_encode --pattern_dirs PATTERN_DIRS [PATTERN_DIRS ...] --tnc_dir TNC_DIR [--n_threads N_THREADS]`

**参数说明：**

- pattern_dirs：存储原始pattern文件的目录，可以同时传入多个
- tnc_dir：tnc编码结果的存储路径
- n_threads：并发数。默认为4

## AutoEncoder encode

会在每个切分目录下生成encode文件夹，生成对应样本使用AE编码后的结果

> command line:
>
> `python -m infer encode --working_dir WORKING_DIR --tnc_dir TNC_DIR --splits SPLITS SPLITS`

**参数说明：**

- working_dir：实验目录，例如上图中`LC015_531_pre_samples`文件夹的路径
- tnc_dir：tnc编码结果的目录，应和tnc_encode中的tnc_dir保持一致
- splits：需要编码的切分。`--splits 1 11`等价于处理1-10共10个切分

## LightGBM train

每个切分的目录下会生成`model/lgbm.pkl`模型文件

> command line:
>
> `python -m infer lightgbm_train --working_dir WORKING_DIR [--external_files [EXTERNAL_FILES [EXTERNAL_FILES ...]]] --splits SPLITS SPLITS `

**参数说明：**

- working_dir：实验目录，同上
- external_files：2/3、3/5的metrics 文件（注意传入顺序）
- splits：运行的切分。同上

## LightGBM test

在每个切分下生成对应样本的概率，存储在每个切分目录下的`proba.tsv`中

> command line:
>
> `python -m infer lightgbm_test --working_dir WORKING_DIR --external_files [EXTERNAL_FILES [EXTERNAL_FILES ...]] --split_file SPLIT_FILE --splits SPLITS SPLITS `

**参数说明：**

- working_dir：实验目录，同上
- external_files：2/3、3/5的metrics 文件（注意传入顺序）
- split_file：选择测试哪些样本（每个切分下split目录下对应的文件）可选`train.tsv`、`test.tsv`等
- splits：运行的切分。同上