# Research__MSSNet

## run mseg for image segmentation

### Dependencies

Install the `mseg` module from [`mseg-api`](https://github.com/mseg-dataset/mseg-api).

### Install the MSeg-Semantic module:

* `mseg_semantic` can be installed as a python package using

        pip install -e /path_to_root_directory_of_the_repo/

Make sure that you can run `python -c "import mseg_semantic; print('hello world')"` in python, and you are good to go!


### MSeg Pre-trained Models

Each model is 528 MB in size. We provide download links and testing results (**single-scale** inference) below:

Abbreviated Dataset Names: VOC = PASCAL VOC, PC = PASCAL Context, WD = WildDash, SN = ScanNet

|    Model                | Training Set    |  Training <br> Taxonomy | VOC <br> mIoU | PC <br> mIoU | CamVid <br> mIoU | WD <br> mIoU | KITTI <br> mIoU | SN <br> mIoU | h. mean | Download <br> Link        |
| :---------------------: | :------------:  | :--------------------:  | :----------:  | :---------------------------: | :--------------: | :----------: | :-------------: | :----------: | :----:  | :--------------: |
| MSeg (1M)               | MSeg train      | Universal               | 70.7          | 42.7                          | 83.3             | 62.0         | 67.0            | 48.2         | 59.2    | [Google Drive](https://drive.google.com/file/d/1g-D6PtXV-UhoIYFcQbiqcXWU2Q2M-zo9/view?usp=sharing) |
| MSeg (3M)-480p               | MSeg <br> train      | Universal         | 76.4 |  45.9 |  81.2 |  62.7 |  68.2 |  49.5 |  61.2  | [Google Drive](https://drive.google.com/file/d/1BeZt6QXLwVQJhOVd_NTnVTmtAO1zJYZ-/view?usp=sharing) |
| MSeg (3M)-720p               | MSeg <br> train      | Universal               | 74.7 |  44.0 |  83.5 |  60.4 |  67.9 |  47.7 |  59.8 | [Google Drive](https://drive.google.com/file/d/1Y9rHOn_8e_qLuOnl4NeOeueU-MXRi3Ft/view?usp=sharing) |
| MSeg (3M)-1080p               | MSeg <br> train      | Universal               | 72.0 |  44.0 |  84.5 |  59.9 |  66.5 |  49.5 |  59.8 | [Google Drive](https://drive.google.com/file/d/1iobea9IW2cWPF6DtM04OvKtDRVNggSb5/view?usp=sharing) |

### run
```
model_name=mseg-3m
model_path=/path/to/downloaded/model/from/google/drive
config=mseg_semantic/config/test/default_config_360_ms.yaml
python -u mseg_semantic/tool/universal_demo.py \
  --config=${config} model_name ${model_name} model_path ${model_path} input_file ${input_file}
```
## run PVKD for point cloud segmentation

### Requirements
- PyTorch >= 1.2 
- yaml
- tqdm
- numba
- Cython
- [torch-scatter](https://github.com/rusty1s/pytorch_scatter)
- [nuScenes-devkit](https://github.com/nutonomy/nuscenes-devkit) (optional for nuScenes)
- [spconv](https://github.com/traveller59/spconv) (tested with spconv==1.2.1 and cuda==10.2)

### run
We take evaluation on the SemanticKITTI test set (single-scan) as example.

1. Download the [pre-trained models](https://drive.google.com/drive/folders/1LyWhVCqMzSVDe44c8ARDp8b94w1ct-tR?usp=sharing) and put them in `./model_load_dir`.

2. Generate predictions on the SemanticKITTI test set.

```
CUDA_VISIBLE_DEVICES=0 python -u test_cyl_sem_tta.py
```

We perform test-time augmentation to boost the performance. The model predictions will be saved in `./out_cyl/test` by default.


3. Convert label number back to the original dataset format:
```
python remap_semantic_labels.py -p out_label/test -s test --inverse
```
## run for calibration

### Requirements
The code has been tested with PyTorch 1.6 and Cuda 10.1

and
```commandline
pip install -r requirements.txt
```
### Train
```commandline
python train.py
```