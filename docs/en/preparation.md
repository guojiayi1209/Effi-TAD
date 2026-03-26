# Installation

**Step 1.** Install PyTorch=2.0.0, Python=3.8.20

```
conda create -n opentad python=3.8.20
source activate opentad
conda install pytorch=2.0.0 torchvision=0.15.0 pytorch-cuda=11.8 -c pytorch -c nvidia
```

**Step 2.** Install mmaction2 for end-to-end training
```
pip install openmim
mim install mmcv==2.0.1
mim install mmaction2==1.1.0
```

**Step 3.** Install OpenTAD
```
git clone https://github.com/guojiayi1209/Effi-TAD.git

pip install -r requirements.txt
```


# Prepare the Annotation and Data

The following table lists the supported datasets and provides links to the corresponding data preparation instructions.

| Dataset                                                    | Description                                                                                   |
| :--------------------------------------------------------- | :-------------------------------------------------------------------------------------------- |
| [ActivityNet](/tools/prepare_data/activitynet/README.md)   | A Large-Scale Video Benchmark for Human Activity Understanding with 19,994 videos.            |
| [THUMOS14](/tools/prepare_data/thumos/README.md)           | Consists of 413 videos with temporal annotations.                                             |
| [Multi-THUMOS](/tools/prepare_data/multi-thumos/README.md) | Dense, multilabel action annotations of THUMOS14.                                             |
| [Charades](/tools/prepare_data/charades/README.md)         | Contains dense-labeled 9,848 annotated videos of daily activities.                            |


## FAQ

1. If you meet `FileNotFoundError: [Errno 2] No such file or directory: 'xxx/missing_files.txt'`
- It means you may need to generate a `missing_files.txt`, which should record the missing features compared to all the videos in the annotation files. You can use `python tools/prepare_data/generate_missing_list.py annotation.json feature_folder` to generate the txt file.
- eg. `python tools/prepare_data/generate_missing_list.py data/fineaction/annotations/annotations_gt.json  data/fineaction/features/fineaction_mae_g`
- In the provided feature from this codebase, we have already included this txt in the zip file.

## Prepare the pretrained VideoMAE checkpoints

Before running the experiments, please download the pretrained VideoMAE model weights (converted from original repo), and put them under the path `./pretrained/`.

|    Model     | Pretrain Dataset | Finetune Dataset |                                           Original Link                                           |                                                                  Converted Checkpoints                                                                   |
| :----------: | :--------------: | :--------------: | :-----------------------------------------------------------------------------------------------: | :------------------------------------------------------------------------------------------------------------------------------------------------------: |
|  VideoMAE-S  |       K400       |       K400       | [Url](https://github.com/MCG-NJU/VideoMAE/blob/main/MODEL_ZOO.md#:~:text=/log/-,checkpoint,-79.0) |                            [Google Drive](https://drive.google.com/file/d/1BH5BZmdImaZesUfqtW23eBGC341Gui1D/view?usp=sharing)                            |
|  VideoMAE-B  |       K400       |       K400       | [Url](https://github.com/MCG-NJU/VideoMAE/blob/main/MODEL_ZOO.md#:~:text=/log/-,checkpoint,-81.5) | [mmaction2](https://download.openmmlab.com/mmaction/v1.0/recognition/videomae/vit-base-p16_videomae-k400-pre_16x4x1_kinetics-400_20221013-860a3cd3.pth)  |
|  VideoMAE-L  |       K400       |       K400       | [Url](https://github.com/MCG-NJU/VideoMAE/blob/main/MODEL_ZOO.md#:~:text=/log/-,checkpoint,-85.2) | [mmaction2](https://download.openmmlab.com/mmaction/v1.0/recognition/videomae/vit-large-p16_videomae-k400-pre_16x4x1_kinetics-400_20221013-229dbb03.pth) |
|  VideoMAE-H  |       K400       |       K400       | [Url](https://github.com/MCG-NJU/VideoMAE/blob/main/MODEL_ZOO.md#:~:text=/log/-,checkpoint,-86.6) |                            [Google Drive](https://drive.google.com/file/d/1wWXs7xpkVkQ2cJnvRVKnWZ86QswZC1UI/view?usp=sharing)                            |
| VideoMAEv2-g |      Hybrid      |       K710       |           [Url](https://github.com/OpenGVLab/VideoMAEv2/blob/master/docs/MODEL_ZOO.md)            |                                                                       Not Provided


## Usage

By default, we use DistributedDataParallel (DDP) both in single GPU and multiple GPU cases for simplicity.

### Training

`torchrun --nnodes={num_node} --nproc_per_node={num_gpu} --rdzv_backend=c10d --rdzv_endpoint=localhost:0 tools/train.py {config}`

- `num_node` is often set as 1 if all gpus are allocated in a single node. `num_gpu` is the number of used GPU.
- `config` is the path of the config file.

Example:

- Training feature-based ActionFormer on 1 GPU.
```bash
torchrun \
    --nnodes=1 \
    --nproc_per_node=1 \
    --rdzv_backend=c10d \
    --rdzv_endpoint=localhost:0 \
    tools/train.py configs/actionformer/thumos_i3d.py
```

- Training end-to-end-based Effi-TAD on 4 GPUs within 1 node.
```bash
torchrun \
    --nnodes=1 \
    --nproc_per_node=4 \
    --rdzv_backend=c10d \
    --rdzv_endpoint=localhost:0 \
    tools/train.py configs/adatad/e2e_anet_videomae_s_adapter_frame768_img160.py
```

Note:
- **GPU number would affect the detection performance in most cases.** Since TAD dataset is small, and the number of ground truth actions per video differs dramatically in different videos. Therefore, the recommended setting for training feature-based TAD is 1 GPU, empirically.
- By default, evaluation is also conducted in the training, based on the argument in the config file. You can disable this, or increase the evaluation interval to speed up the training. 

### Inference and Evaluation

`torchrun --nnodes={num_node} --nproc_per_node={num_gpu} --rdzv_backend=c10d --rdzv_endpoint=localhost:0 tools/test.py {config} --checkpoint {path}`

- if `checkpoint` is not specified, the `best.pth` in the config's result folder will be used.


Example:

- Inference and Evaluate ActionFormer on 1 GPU.
```bash
torchrun \
    --nnodes=1 \
    --nproc_per_node=1 \
    --rdzv_backend=c10d \
    --rdzv_endpoint=localhost:0 \
    tools/test.py \
    configs/actionformer/thumos_i3d.py \
    --checkpoint exps/thumos/.../gpu1_id0/checkpoint/epoch_34.pth
```