# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.
# cd /clipbert; source setup.sh; python src/tasks/run_video_qa.py --config src/configs/sqa_video_base_resnet50.json --output_dir /storage

TXT_DB=$1
IMG_DIR=$2
OUTPUT=$3
PRETRAIN_DIR=$4

# if [ -z $CUDA_VISIBLE_DEVICES ]; then
#     CUDA_VISIBLE_DEVICES='all'
# fi
# 
# docker run --gpus '"'device=$CUDA_VISIBLE_DEVICES'"' --ipc=host --rm -it \
#     --mount src=$(pwd),dst=/clipbert,type=bind \
#     --mount src=$OUTPUT,dst=/storage,type=bind \
#     --mount src=$PRETRAIN_DIR,dst=/pretrain,type=bind,readonly \
#     --mount src=$TXT_DB,dst=/txt,type=bind,readonly \
#     --mount src=$IMG_DIR,dst=/img,type=bind,readonly \
#     -e NVIDIA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES \
#     -w /clipbert jayleicn/clipbert:latest \
#     bash -c "source /clipbert/setup.sh && bash" \

module load singularity
singularity exec --net --fakeroot --writable --nv \
	--bind $(pwd):/clipbert \
	--bind $OUTPUT:/storage \
	--bind $PRETRAIN_DIR:/pretrain \
	--bind $TXT_DB:/txt \
	--bind $IMG_DIR:/img \
	/scratch/ml/maxiaojian/clipbert/clipbert_singularity_new \
    bash -c "cd /clipbert; source setup.sh; python src/tasks/run_video_qa.py --config src/configs/sqa_video_base_resnet50.json --output_dir /storage"
