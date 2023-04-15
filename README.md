## HRSTNet
This repository contains the code for the paper High-Resolution Swin Transformer for Automatic Medical Image Segmentation.

### Prerequisites
* Python 3.8
* Pytorch 1.8.1
* monai 0.9.0rc2+2.gbbc628d9 (results in paper submitted to hindawi)
* monai 0.9.1 (new results)
* nnunet pip install nnunet
* pip install axial-attention==0.5.0
* pip install mmcv-full
* pip install einops
* pip install SimpleITK
* pip install tensorboardX
* pip install omegaconf
* pip install fvcore

### Environments
* Ubuntu 18.04
* cuda 10.2
* cudnn 8.1.1

### Training
* Modify the parameters about datasets path in the `configs/dataset_path.yaml` file.

#### BraTS 2021
1. The config file for training BraTS 2021 dataset is `configs/hr_trans_brats_2021_seg.yaml`.
2. The following parameters in the above config file should be specified before training: 
   * MODEL.HR_TRANS.STAGE_NUM: the number of stages in the HRSTNet.
   * DATALOADER.TRAIN_WORKERS: the number of workers of pytorch dataloader for training. This parameter should be set according to your computer configuration.
   * SOLVER.EPOCHS: training epochs.
   * OUTPUT_DIR: the path to save the training log and model files.
   * INPUT.RAND_CROP.SAMPLES: the number of random crop samples in each batch, and the real batch size equals INPUT.RAND_CROP.SAMPLES*SOLVER.BATCH_SIZE.
3. Using the file `tools_train.py` to train the model.
4. Modify the following parameters i the `tools_train.py` file before training:
   * --config_file: the path to the config file that is used to train model.
   * --num-gpus: gpu count used to train model.

#### MSD
1. The config file for training MSD dataset is `configs/hr_trans_liver_seg.yaml`.
2. The following parameters in the above config file should be specified before training: 
   * MODEL.HR_TRANS.STAGE_NUM: the number of stages in the HRSTNet.
   * DATALOADER.TRAIN_WORKERS: the number of workers of pytorch dataloader for training. This parameter should be set according to your computer configuration.
   * SOLVER.EPOCHS: training epochs.
   * OUTPUT_DIR: the path to save the training log and model files.
   * INPUT.RAND_CROP.SAMPLES: the number of random crop samples in each batch, and the real batch size equals INPUT.RAND_CROP.SAMPLES*SOLVER.BATCH_SIZE.
3. Using the file `tools_train.py` to train the model.
4. Modify the following parameters i the `tools_train.py` file before training:
   * --config_file: the path to the config file that is used to train model.
   * --num-gpus: gpu count used to train model.

#### Abdomen
1. The config file for training MSD dataset is `configs/hr_trans_abdomen_seg.yaml`.
2. The following parameters in the above config file should be specified before training: 
   * MODEL.HR_TRANS.STAGE_NUM: the number of stages in the HRSTNet.
   * DATALOADER.TRAIN_WORKERS: the number of workers of pytorch dataloader for training. This parameter should be set according to your computer configuration.
   * SOLVER.EPOCHS: training epochs.
   * OUTPUT_DIR: the path to save the training log and model files.
   * INPUT.RAND_CROP.SAMPLES: the number of random crop samples in each batch, and the real batch size equals INPUT.RAND_CROP.SAMPLES*SOLVER.BATCH_SIZE.
3. Using the file `tools_train.py` to train the model.
4. Modify the following parameters i the `tools_train.py` file before training:
   * --config_file: the path to the config file that is used to train model.
   * --num-gpus: gpu count used to train model.


### Evaluation
After training the models by using the `tools_train.py` file, the performance of models are evaluated by using the `tools_inference.py` file.
If running the `evaluate_*.py` encounter the following error:
`ITK ERROR: ITK only supports orthonormal direction cosines.  No orthonormal definition found!`
Running the code `evaluate/fix_simpleitk_read_error.py` file to fix the wrong file, and setting the parameter `pred_file_path` to the wrong file path, and the `original_img_path` to the corresponding image path.

#### BraTS 2021
If the model is trained by using the `vt_unet` preprocessing method, the model is evaluated by the following method (taking `hrstnet` as example).
1. modify the parameters `MODEL.WEIGHTS`, `OUTPUT_DIR`, and `MODE` in the `configs/hr_trans_brats_2021_seg.yaml`.
2. set the parameter `space=brats_2021_vt_unet`, and `config_file=confgs/configs/hr_trans_brats_2021_seg.yaml` in the `tools_inference.py` file.
3. Running the `tools_inference.py` file, and the segmentation results will be generated in the folder `OUTPUT_DIR/seg_results`.
4. The segmentation masks in folder `OUTPUT_DIR/seg_results` can be used to visualize in the 3D Slicer software.
5. Running the code `evaluate/evaluate_brats_vt_unet.py`, and setting the parameters `inferts_path`, and `path` to `OUTPUT_DIR/seg_results`, and `ground_truth_path`, respectively.
6. The evaluation results will appear in `OUTPUT_DIR/seg_results/dice_pre.txt`.

#### MSD
Taking the Spleen dataset from MSD as an example.
1. modify the parameters `MODEL.WEIGHTS`, `OUTPUT_DIR`, and `MODE` in the `configs/hr_spleen_seg.yaml`.
2. set the parameter `space=original_msd`, and `config_file=confgs/configs/hr_trans_spleen_seg.yaml` in the `tools_inference.py` file.
3. Running the `tools_inference.py` file, and the segmentation results will be generated in the folder `OUTPUT_DIR/seg_results/`.
4. The segmentation masks in folder `OUTPUT_DIR/seg_results` can be used to visualize in the 3D Slicer software.
5. Running the code `evaluate/evaluate_msd.py`, and setting the parameters `pred_path`, `MSD_TYPE`, `CATEGORIES`, `MODE` and `gt_path` to `OUTPUT_DIR/seg_results`, `Spleen`, `2`, `VALIDATE` and `ground_truth_path`, respectively.
6. The evaluation results will appear in `OUTPUT_DIR/seg_results/dice_pred.txt`.
7. Running the code `evaluate/evaluate_msd.py`, and setting the parameter `MODE` to `VAL` to check does all the generated files are correct.

#### Abdomen
1. modify the parameters `MODEL.WEIGHTS`, `OUTPUT_DIR`, `DATASETS.TEST_TYPE`, and `MODE` in the `configs/hr_trans_abdomen_seg.yaml`.
2. set the parameter `space=original_abdomen`, and `config_file=confgs/configs/hr_trans_abdomen_seg.yaml` in the `tools_inference.py` file.
3. Running the `tools_inference.py` file, and the segmentation results will be generated in the folder `OUTPUT_DIR/seg_results/`.
4. Utilizing the `evaluate/fix_aliginment_error.py` file to modify the origin and direction of saved CT segmentation file, otherwise the segmentation mask can not display correctly. Changing the `pred_img_folder`, and the `gt_img_folder` parameters.
```python
pred_img_folder = "/home/ljm/Fdisk_A/train_outputs/train_output_medical_2022_8/hrstnet/abdomen_seg_hrstnet_stages_4/seg_results/"
gt_img_folder = "/home/ljm/Fdisk_A/train_datasets/train_datasets_medical/2015_Segmentation_Cranial Vault Challenge/Abdomen/RawData/Training/img/"
```
5. The segmentation masks in folder `OUTPUT_DIR/seg_results` can be used to visualize in the 3D Slicer software.
6. Running the code `evaluate/evaluate_abdomen.py`, and setting the parameters `pred_path`, and `gt_path` to `OUTPUT_DIR/seg_results`, `Abdomen/RawData/Training/label/`, respectively.
7. The evaluation results will appear in `OUTPUT_DIR/seg_results/dice_pred.txt`.


### Other files
* `tools_visualize.py`: is used to visualize the segmentation results.
* `tools_flops.py`: is used to count the flops of models.


### Acknowledge
1. [detectron2](https://github.com/facebookresearch/detectron2)
2. [MONAI](https://github.com/Project-MONAI/MONAI)
3. [UNETR](https://github.com/Project-MONAI/research-contributions)
4. [VT-UNET](https://github.com/himashi92/VT-UNet)
5. [nnUnet](https://github.com/MIC-DKFZ/nnUNet)
6. [Extending nn-UNet](https://github.com/rixez/Brats21_KAIST_MRI_Lab)
7. [Optimized UNet](https://github.com/NVIDIA/DeepLearningExamples/tree/master/PyTorch/Segmentation/nnUNet)


### Citation
If you find this project useful for your research, please cite our paper:
```bibtex
@Article{s23073420,
    AUTHOR = {Wei, Chen and Ren, Shenghan and Guo, Kaitai and Hu, Haihong and Liang, Jimin},
    TITLE = {High-Resolution Swin Transformer for Automatic Medical Image Segmentation},
    JOURNAL = {Sensors},
    VOLUME = {23},
    YEAR = {2023},
    NUMBER = {7},
    ARTICLE-NUMBER = {3420},
    URL = {https://www.mdpi.com/1424-8220/23/7/3420},
    ISSN = {1424-8220},
    DOI = {10.3390/s23073420}
}
```


### Contact
Chen Wei

email: weichen@xupt.edu.cn

