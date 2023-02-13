# Running MCAN on SQA3D

## Data preparation for BEV pictures

1. Download [Blender](https://www.blender.org/download/)

2. Download the ScanNetV2 dataset and put (or link) `scans/` under (or to) `../assets/data/scannet/scans/` (Please follow the [ScanNet Instructions](../assets/data/scannet/README.md) for downloading the ScanNet dataset).

3. Use the following command to render the input image for MCAN model
```shell
cd ../utils
blender -b file.blend --python mesh2img.py
``` 
For convenience, you can download the images rendered by us from [here](https://zenodo.org/record/7544818/files/bird.zip?download=1)

4. Download the pretrained vision backbones and other files from [here](https://drive.google.com/file/d/1pxmUxkk5t8Bg_cS_jdaQgugCqYddZInE/view?usp=sharing) and extract them to `./cache`
5. Download the preprocessed [SpaCy embedding](en_vectors_web_lg) and then run
```shell
pip install path/en_core_web_lg-1.2.0.tar.gz
```

## Training
```python
train_sqa.py --config-file train_sqa_mcan.yaml
```

## Evaluation
```python
train_sqa.py --test_only --config-file train_sqa_mcan.yaml --test_model <model_path>
```
<model_path> corresponds to the path to the model.

## Pretrained models
- Pretrained models can be downloaded [here](https://drive.google.com/drive/folders/1WJlvLUslAOwe846oJ1W4kpmck_SlkPUR?usp=share_link). The correspondence between the models and the results in the paper is as follows
    | models                                   |  Model in the paper  | results |
    |------------------------------------------|----------------------|---------|
    | `MCAN.pth`                               | `MCAN`               |  43.42  |
Note that due to the slight change of codebase, the results evaluated is slightly higher than presented in the paper(around 1%).

## Acknowledgements
We would like to thank [MCAN](https://github.com/MILVLG/mcan-vqa) and [RelViT](https://github.com/NVlabs/RelViT) for their useful code bases.
