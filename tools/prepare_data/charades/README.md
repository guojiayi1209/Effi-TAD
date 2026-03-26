# Data Preparation for Charades

## Download Annotations

You can download the annotation by
```bash
bash download_annotation.sh
```
The annotation should be in `data/charades/annotations/`.


## Download Pre-extracted Features

Please put the downloaded feature under the path: `data/charades/features/`.


We provide the following pre-extracted features for Charades:

|  Feature   |                                                Url                                                 |  Backbone  |                                                   Feature Extraction Setting                                                   |
| :--------: | :------------------------------------------------------------------------------------------------: | :--------: | :----------------------------------------------------------------------------------------------------------------------------: |
|  vgg-rgb   | [Google Drive](https://drive.google.com/file/d/1qrHK5Rcktxh0eoWd-BMyWN-XbQ9XwCkc/view?usp=sharing) |   VGG-16   |            24fps, snippet_stride=4, converted from [official website](https://prior.allenai.org/projects/charades)             |
|  vgg-flow  | [Google Drive](https://drive.google.com/file/d/1kvuIBsMZzWKBPCVhQNmX0iBkSHIQF65T/view?usp=sharing) |   VGG-16   |            24fps, snippet_stride=4, converted from [official website](https://prior.allenai.org/projects/charades)             |
|  i3d-rgb   | [Google Drive](https://drive.google.com/file/d/1NNXWLi4O0P_TAiKaPCr_w2XppYiZZNwQ/view?usp=sharing) |    I3D     | 24fps, snippet_stride=8, converted from [here](https://github.com/Xun-Yang/Causal_Video_Moment_Retrieval/blob/main/DATASET.md) |
| videomae-l | [Google Drive](https://drive.google.com/file/d/1eL_xpkjwbyRCNRKSL1bhq95kJc_FrKjq/view?usp=sharing) | VideoMAE-L |                                   30fps, snippet_stride=4, clip_length=16, frame_interval=1                                    |


## Download Raw Videos

Please put the downloaded video under the path: `data/charades/raw_data/`.

You can download the raw video from [official website](https://prior.allenai.org/projects/charades), and then convert it to 30 fps.

We also provide the converted videos with 30 fps at [Google Drive](https://drive.google.com/file/d/10NiCMo5KJcTo0nCr2_hUYC1V4nzP06Eq/view?usp=sharing).
