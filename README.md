SQA3D
===
<p align="center"><img width="540" src="./assets/overview.JPG"></p>

This repository hosts the code for the paper:

[SQA3D: Situated Question Answering in 3D Scenes](https://arxiv.org/pdf/2210.07474.pdf)

by [Xiaojian Ma*](http://web.cs.ucla.edu/~xm), [Silong Yong*](https://silongyong.github.io/), [Zilong Zheng](https://zilongzheng.github.io/), [Qing Li](https://liqing-ustc.github.io/), [Yitao Liang](https://web.cs.ucla.edu/~yliang/), [Song-Chun Zhu](http://www.stat.ucla.edu/~sczhu/) and [Siyuan Huang](https://siyuanhuang.com/)

[arXiv](http://arxiv.org/abs/2205.13803) | [Benchmark]()

## Abstract

We propose a new task to benchmark scene understanding of embodied agents: Situated Question Answering in 3D Scenes (**SQA3D**). Given a scene context(e.g., 3D scan), SQA3D requires the tested agent to first understand its situation(position, orientation, etc.) in the 3D scene as described by text, then reason about its surrounding environment and answer a question under that situation. Based upon 650 scenes from ScanNet, we provide a dataset centered around 6.8k unique situations, along with 20.4k descriptions and 33.4k diverse reasoning questions for these situations. These questions examine a wide spectrum of reasoning capabilities for an intelligent agent, ranging from spatial relation comprehension to commonsense understanding, navigation, and multi-hop reasoning. SQA3D imposes a significant challenge to current multi-modal especially 3D reasoning models. We evaluate various state-of-the-art approaches and find that the best one only achieves an overall score of **47.20%**, while amateur human participants can reach **90.06%**. We believe SQA3D could facilitate future embodied AI research with stronger situation understanding and reasoning capability.

## Installation

- Install PyTorch:
    ```shell
    conda install pytorch torchvision torchaudio cudatoolkit=11.3 -c pytorch
    ```

- Install the necessary packages with `requirements.txt`:
    ```shell
    pip install -r requirements.txt
    ```

The code has been tested with Python 3.9, PyTorch 1.12.0 and CUDA 11.3 on Ubuntu 20.04

##  Data Preparation

Please refer to [data preparation](assets/dataset.md)

## Training

Please refer to [ScanQA data](../ScanQA/README.md), [MCAN data](../MCAN/README.md), [ClipBERT data](../ClipBERT/README.md) for details.

### Note

When running corresponding model, please
```shell
PATH = ./MODEL_YOU_WANT_TO_RUN
cd PATH
```

## License

Please check the LICENSE file. This work may be used non-commercially, meaning for research or evaluation purposes only. For business inquiries, please contact researchinquiries@nvidia.com.

Please note that this repo only hosts the code of SQA3D, the SQA3D datasest is released by Caltech and shall not be viewed as part of the source code.

## Citation
If you find our work helpful for your research. Please consider citing our paper.
```bibtex
@article{ma2022sqa3d,
  title={SQA3D: Situated Question Answering in 3D Scenes},
  author={Ma, Xiaojian and Yong, Silong and Zheng, Zilong and Li, Qing and Liang, Yitao and Zhu, Song-Chun and Huang, Siyuan},
  journal={arXiv preprint arXiv:2210.07474},
  year={2022}
}
```