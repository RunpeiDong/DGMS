# Differentiable Gaussian Mixture Weight Sharing Network Quantization, ICML 2022 Spotlight
> [**Finding the Task-Optimal Low-Bit Sub-Distribution in Deep Neural Networks**](https://proceedings.mlr.press/v162/dong22a.html), ICML 2022<br>
> [Runpei Dong](https://runpeidong.com/)\*, [Zhanhong Tan](https://www.zhanhongtan.com/)\*, [Mengdi Wu](), [Linfeng Zhang](https://scholar.google.com.hk/citations?user=AK9VF30AAAAJ&hl=en), and [Kaisheng Ma](http://group.iiis.tsinghua.edu.cn/~maks/leader.html) <br>

Created by [Runpei Dong](https://runpeidong.com/)\*, [Zhanhong Tan](https://www.zhanhongtan.com/)\*, [Mengdi Wu](https://scholar.google.com.hk/citations?user=F9EN5zgAAAAJ&hl=en&oi=sra), [Linfeng Zhang](http://group.iiis.tsinghua.edu.cn/~maks/linfeng/index.html), and [Kaisheng Ma](http://group.iiis.tsinghua.edu.cn/~maks/leader.html).

[PMLR](https://proceedings.mlr.press/v162/dong22a.html) | [arXiv](https://arxiv.org/abs/2112.15139) | [Models](https://drive.google.com/drive/folders/1rQJLAbP8gb5ZIUyIjEVHof0euyhsVGu4?usp=sharing)

This repository contains the code release of the paper **Finding the Task-Optimal Low-Bit Sub-Distribution in Deep Neural Networks** (ICML 2022).


## Installation

Our code works with Python 3.8.3. we recommend to use [Anaconda](https://www.anaconda.com/), and you can install the dependencies by running:

```shell
$ python3 -m venv env
$ source env/bin/activate
(env) $ python3 -m pip install -r requirements.txt
```
## How to Run

The main procedures are written in script `main.py`, please run the following command for instructions:

```shell
$ python main.py -h
```

### Datasets

Before running the code, you can specify the path for datasets in `config.py`, or you can specify it by `--train-dir` and `--val-dir`.

### Training on ImageNet

We have provided a simple SHELL script to train a 4-bit `ResNet-18` with `DGMS`. Run:

```shell
$ sh tools/train_imgnet.sh
```

### Inference on ImageNet

To inference compressed models on ImageNet, you only need to follow 2 steps:

* **Step-1**: Download the checkpoints released on [Google Drive](https://drive.google.com/drive/folders/1rQJLAbP8gb5ZIUyIjEVHof0euyhsVGu4?usp=sharing).

* **Step-2**: Run the inference SHELL script we provide:

  ```shell
  $ sh tools/validation.sh
  ```

## Q-SIMD

The [TVM](https://github.com/apache/tvm) based Q-SIMD codes can be download from [Google Drive](https://drive.google.com/file/d/1hGeXXdHetGKZKSd4dp7xTSRxjWXgPkjc/view?usp=sharing).

## Citation

If you find our work useful in your research, please consider citing:

```tex
@inproceedings{dong2021finding,
  title={Finding the Task-Optimal Low-Bit Sub-Distribution in Deep Neural Networks},
  author={Dong, Runpei and Tan, Zhanhong and Wu, Mengdi and Zhang, Linfeng and Ma, Kaisheng},
  booktitle={Proceedings of the International Conference on Machine Learning (ICML)},
  year={2022}
}
```

## License

DGMS is released under the Apache 2.0 license.  See the [LICENSE](./LICENSE) file for more details.
