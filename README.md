# DGMS

This is the code of the paper *Finding the Task-Optimal Low-Bit Sub-Distribution in Deep Neural Networks*.

## Installation

Our code works with Python 3.8.3 and the following packages are required:

*  [PyTorch](https://pytorch.org/) 1.6.0

* Torchvison 0.7.0

* TensorboardX
* Pillow
* graphviz
* pydot
* kmeans-pytorch

We recommend to use [Anaconda](https://www.anaconda.com/), and you can install the dependencies by running:

```shell
$ python3 -m venv env
$ source env/bin/activate
(env) $ python3 -m pip install -r requirements.txt
```

### How to Run

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

## Citation

```tex
@article{dong2021finding,
      title={Finding the Task-Optimal Low-Bit Sub-Distribution in Deep Neural Networks}, 
      author={Runpei Dong and Zhanhong Tan and Mengdi Wu and Linfeng Zhang and Kaisheng Ma},
      year={2021},
      eprint={2112.15139},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
```

## License

DGMS is released under the [Apache 2.0 license](./LICENSE).  See the LICENSE file for more details.
