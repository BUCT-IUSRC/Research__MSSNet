Point-to-Voxel Knowledge Distillation for LiDAR Semantic Segmentation (CVPR 2022)

Our model achieves state-of-the-art performance on three challenges, i.e., ranks **1st** in [Waymo 3D Semantic Segmentation Challenge](https://waymo.com/open/challenges/2022/3d-semantic-segmentation/) (the "Cylinder3D" and "Offboard_SemSeg" entries, May 2022), ranks **1st** in [SemanticKITTI LiDAR Semantic Segmentation Challenge](https://competitions.codalab.org/competitions/20331#results) (single-scan, the "Point-Voxel-KD" entry, Jun 2022), ranks **2nd** in [SemanticKITTI LiDAR Semantic Segmentation Challenge](https://competitions.codalab.org/competitions/20331#results) (multi-scan, the "PVKD" entry, Dec 2021). Do not hesitate to use our trained models!

## News

- **2022-11** [NEW:fire:] Some useful training tips have been provided.

- **2022-11** The distillation codes and some training tips will be released after CVPR DDL.

- **2022-7** We provide a trained model of [CENet](https://github.com/huixiancheng/CENet), a range-image-based LiDAR segmentation method. The reproduced performance is much higher than the reported value! 

- **2022-6** Our method ranks **1st** in [SemanticKITTI LiDAR Semantic Segmentation Challenge](https://competitions.codalab.org/competitions/20331#results) (single-scan, the "Point-Voxel-KD" entity)
<p align="center">
   <img src="./img/semantickitti_single_scan.PNG" width="30%"> 
</p>

- **2022-5** Our method ranks **1st** in [Waymo 3D Semantic Segmentation Challenge](https://waymo.com/open/challenges/2022/3d-semantic-segmentation/) (the "Cylinder3D" and "Offboard_SemSeg" entities)
<p align="center">
   <img src="./img/waymo.PNG" width="30%"> 
</p>

## Installation

### Requirements
- PyTorch >= 1.2 
- yaml
- tqdm
- numba
- Cython
- [torch-scatter](https://github.com/rusty1s/pytorch_scatter)
- [nuScenes-devkit](https://github.com/nutonomy/nuscenes-devkit) (optional for nuScenes)
- [spconv](https://github.com/traveller59/spconv) (tested with spconv==1.2.1 and cuda==10.2)

## Data Preparation

### SemanticKITTI
```
./
├── 
├── ...
└── path_to_data_shown_in_config/
    ├──sequences
        ├── 00/           
        │   ├── velodyne/	
        |   |	├── 000000.bin
        |   |	├── 000001.bin
        |   |	└── ...
        │   └── labels/ 
        |       ├── 000000.label
        |       ├── 000001.label
        |       └── ...
        ├── 08/ # for validation
        ├── 11/ # 11-21 for testing
        └── 21/
	    └── ...
```

### nuScenes
```
./
├── 
├── ...
└── path_to_data_shown_in_config/
		├──v1.0-trainval
		├──v1.0-test
		├──samples
		├──sweeps
		├──maps

```

### Waymo
```
./
├── 
├── ...
└── path_to_data_shown_in_config/
		├──first_return
		├──second_return

```

## Test
We take evaluation on the SemanticKITTI test set (single-scan) as example.

1. Download the [pre-trained models](https://drive.google.com/drive/folders/1LyWhVCqMzSVDe44c8ARDp8b94w1ct-tR?usp=sharing) and put them in `./model_load_dir`.

2. Generate predictions on the SemanticKITTI test set.

```
CUDA_VISIBLE_DEVICES=0 python -u test_cyl_sem_tta.py
```

We perform test-time augmentation to boost the performance. The model predictions will be saved in `./out_cyl/test` by default.


3. Convert label number back to the original dataset format before submitting:
```
python remap_semantic_labels.py -p out_label/test -s test --inverse
cd out_label/test
zip -r out_label.zip sequences/
```

4. Upload out_cyl.zip to the [SemanticKITTI online server](https://competitions.codalab.org/competitions/20331#participate).

## Train

```
CUDA_VISIBLE_DEVICES=0 python -u train_cyl_sem.py
```

Remember to change the `imageset` of `val_data_loader` to `val`, `return_test` of `dataset_params` to `False` in `semantickitti.yaml`. Currently, we only support vanilla training.


## Useful Training Tips
1. Finetuning.

You can finetune the model using both train and val sets as well as a smaller learning rate (1/3 or 1/4 of the original learning rate).

2. Model ensemble. 

You can use models of different epochs as an ensemble. Different models can also be taken as an ensemble, e.g., SPVCNN and Cylinder3D.

3. Semi-supervised learning. 

You can follow [GuidedContrast](https://arxiv.org/abs/2110.08188) to use pseudo labels of the test set to complement the original training set. (**DO NOT** use it in the supervised training. It can only be used in the semi-supervised setting to prove the value of the proposed semi-supervised algorithm.)

4. More data augmentations. 

You can use [LaserMix](https://arxiv.org/abs/2207.00026), [Instance Augmentation](https://github.com/edwardzhou130/Panoptic-PolarNet/blob/main/dataloader/instance_augmentation.py) and [PolarMix](https://arxiv.org/abs/2208.00223) to increase the diversity of training samples.

5. Knowledge distillation (KD).

You can refer to [CRD](https://github.com/HobbitLong/RepDistiller) to apply KD to boost the performance of LiDAR segmentation models. We will release a more efficient and effective version of the PVKD algorithm soon.

6. Using more inputs.

In addition to the (x, y, z), you can also use the intensity, range, azimuth, inclination and elongation as additional inputs. Remember to normalize these input signals if necessary. Tanh function is a good normalizer in some cases.

7. Increasing the model size.

You can either increase the width (more channels) or the depth (more layers) of the model to boost the performance.

8. Test time augmentation (TTA).

You can use more augmentations (flipping, rotation, scaling, translation) in TTA to boost the performance. A proper combination of them is vital to the final performance.

## Performance

Abbreviation:

cyl: Cylinder3D, sem: SemanticKITTI, nusc: nuScenes, ms: multi-scan task, tta: test-time augmentation,

1.5x: channel expansion ratio, 72_4: performance (mIoU), 64x512: resolution of the range image

1. SemanticKITTI test set (single-scan):

|Model|Reported|Reproduced|Gain|Weight|
|:---:|:---:|:---:|:---:|:---:|
|SPVNAS|66.4%|71.4%|**5.0%**|--|
|Cylinder3D_1.5x|--|**72.4%**|--|[cyl_sem_1.5x_72_4.pt](https://drive.google.com/drive/folders/1LyWhVCqMzSVDe44c8ARDp8b94w1ct-tR?usp=sharing)|
|Cylinder3D|68.9%|71.8%|**2.9%**|[cyl_sem_1.0x_71_8.pt](https://drive.google.com/drive/folders/1LyWhVCqMzSVDe44c8ARDp8b94w1ct-tR?usp=sharing)|
|Cylinder3D_0.5x|71.2%|71.4%|0.2%|[cyl_sem_0.5x_71_4.pt](https://drive.google.com/drive/folders/1LyWhVCqMzSVDe44c8ARDp8b94w1ct-tR?usp=sharing)|
|CENet_1.0x|64.7%|67.6%|2.9%|[CENet_64x512_67_6](https://drive.google.com/drive/folders/1LyWhVCqMzSVDe44c8ARDp8b94w1ct-tR?usp=sharing)|

2. SemanticKITTI test set (multi-scan):

|Model|Reported|Reproduced|Gain|Weight|
|:---:|:---:|:---:|:---:|:---:|
|Cylinder3D|52.5%|--|--|--|
|Cylinder3D_0.5x|58.2%|58.4%|0.2%|[cyl_sem_ms_0.5x_58_4.pt](https://drive.google.com/drive/folders/1LyWhVCqMzSVDe44c8ARDp8b94w1ct-tR?usp=sharing)|

3. Waymo test set:

|Model|Reported|Reproduced|Gain|Weight|
|:---:|:---:|:---:|:---:|:---:|
|Cylinder3D|71.18%|71.18%|--|--|
|Cylinder3D_0.5x|--|--|--|--|

4. nuScenes val set:

|Model|Reported|Reproduced|Gain|Weight|
|:---:|:---:|:---:|:---:|:---:|
|Cylinder3D|76.1%|--|--|--|
|Cylinder3D_0.5x|76.0%|76.15%|0.15%|[cyl_nusc_0.5x_76_15.pt](https://drive.google.com/drive/folders/1LyWhVCqMzSVDe44c8ARDp8b94w1ct-tR?usp=sharing)|

## Citation
If you use the codes, please consider citing the following publications:
```
@inproceedings{pvkd,
    title     = {Point-to-Voxel Knowledge Distillation for LiDAR Semantic Segmentation},
    author    = {Hou, Yuenan and Zhu, Xinge and Ma, Yuexin and Loy, Chen Change and Li, Yikang},
    booktitle = {IEEE Conference on Computer Vision and Pattern Recognition},
    pages     = {8479-8488}
    year      = {2022},
}

@inproceedings{cylinder3d,
    title={Cylindrical and Asymmetrical 3D Convolution Networks for LiDAR Segmentation},
    author={Zhu, Xinge and Zhou, Hui and Wang, Tai and Hong, Fangzhou and Ma, Yuexin and Li, Wei and Li, Hongsheng and Lin, Dahua},
    booktitle={IEEE Conference on Computer Vision and Pattern Recognition},
    pages={9939--9948},
    year={2021}
}

@article{cylinder3d-tpami,
    title={Cylindrical and Asymmetrical 3D Convolution Networks for LiDAR-based Perception},
    author={Zhu, Xinge and Zhou, Hui and Wang, Tai and Hong, Fangzhou and Li, Wei and Ma, Yuexin and Li, Hongsheng and Yang, Ruigang and Lin, Dahua},
    journal={IEEE Transactions on Pattern Analysis and Machine Intelligence},
    year={2021},
    publisher={IEEE}
}
```

## Acknowledgements
This repo is built upon the awesome [Cylinder3D](https://github.com/xinge008/Cylinder3D).
