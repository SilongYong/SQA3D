# Running ScanQA (model) on SQA3D

## Data preparation for 3D Scans

Since this code is based on [ScanRefer](https://github.com/daveredrum/ScanRefer) and [ScanQA](https://github.com/ATR-DBI/ScanQA), you can use the same 3D features. Please also refer to the ScanRefer and ScanQA data preparation.

1. Use the following command to transform data into the format ScanQA model needs.
```shell
cd ../utils
python sqa_data_2_ScanQA.py
``` 

    ### Dataset format
    ```shell
    "scene_id": [ScanNet scene id, e.g. "scene0000_00"],
    "object_id": [], 
    "object_names": [],
    "question_id": [...],
    "question": [...],
    "answers": [...],
    ```

2. Download the preprocessed [GLoVE embedding](http://kaldir.vc.in.tum.de/glove.p) and put them under `../data/`.
3. Download the ScanNetV2 dataset and put (or link) `scans/` under (or to) `data/scannet/scans/` (Please follow the [ScanNet Instructions](data/scannet/README.md) for downloading the ScanNet dataset).
4. Pre-process ScanNet data. A folder named `scannet_data/` will be generated under `data/scannet/` after running the following command:
    ```shell
    cd data/scannet/
    python batch_load_scannet_data.py
    ```

5. (Optional) Pre-process the multiview features from ENet. 

    a. Download [the ENet pretrained weights](http://kaldir.vc.in.tum.de/ScanRefer/scannetv2_enet.pth) and put it under `data/`
    
    b. Download and unzip [the extracted ScanNet frames](http://kaldir.vc.in.tum.de/3dsis/scannet_train_images.zip) under `data/`

    c. Change the data paths in `config.py` marked with __TODO__ accordingly.

    d. Extract the ENet features:
    ```shell
    python scripts/compute_multiview_features.py
    ```

    e. Project ENet features from ScanNet frames to point clouds:
    ```shell
    python scripts/project_multiview_features.py --maxpool
    ```

## Training
- `scripts for training` and the models we evaluated in the paper can be found below
    | `scripts for training`                  |  Model in the paper  |
    |-----------------------------------------|----------------------|
    | `wo3d.sh`                               | `Blind test`         |
    | `wos.sh`                                | `ScanQA (w/o s_txt)` |
    | `full.sh`                               | `ScanQA`             |
    | `auxi.sh`                               | `ScanQA + aux. task` |

## Evaluation
- Evaluation of trained ScanQA models with the val dataset:

  ```shell
  python scripts/test.py --ckpt <folder_name> <--option>
  ```

  <folder_name> corresponds to the folder under outputs/ with the timestamp + <tag_name>.
  <--option> corresponds to the option used when training

## Acknowledgements
We would like to thank [ScanQA](https://github.com/ATR-DBI/ScanQA) for the useful code base.
