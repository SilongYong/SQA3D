# Running MCAN on SQA3D

## Data preparation

1. Download [Blender](https://www.blender.org/download/)

2. Download the ScanNetV2 dataset and put (or link) `scans/` under (or to) `../assets/data/scannet/scans/` (Please follow the [ScanNet Instructions](../assets/data/scannet/README.md) for downloading the ScanNet dataset).

3. Use the following command to render the input image for MCAN model
```shell
cd ../utils
blender -b file.blend --python mesh2img.py
``` 
For convenience, you can download the images rendered by us from [here]()

4. Download the pretrained vision backbones and other files from [here](https://drive.google.com/file/d/1pxmUxkk5t8Bg_cS_jdaQgugCqYddZInE/view?usp=sharing) and extract them to `./cache`
5. Download the preprocessed [SpaCy embedding](en_vectors_web_lg) and then run
```shell
pip install path/en_core_web_lg-1.2.0.tar.gz
```

## Training
```python
train_sqa.py --config-file train_sqa_mcan.yaml
```

# Acknowledgements
We would like to thank [MCAN](https://github.com/MILVLG/mcan-vqa) and [RelViT](https://github.com/NVlabs/RelViT) for their useful code bases.
