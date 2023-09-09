# hypunet

This repo implements hypernetworks for parametrisation of the nnUNet library. Because of this, the majority of the syntax and documentation will follow exactly from nnUNet, however with 'nnunet' swapped out for 'hypunet' on commands. For this reason, we recommend referring to the [nnUNet (v1) Documentation](https://github.com/MIC-DKFZ/nnUNet/tree/nnunetv1).

## Installation

The library may be installed with the command:

`pip install git+https://github.com/liamchalcroft/hypernets`

## Command set-up



## Data pre-processing



## Pre-training

### VICReg

#### Baseline
`hypunet_train 3d_fullres VICRegTrainer 002 all -n baseline-s25-v25-c0.25 -p hypunetPlans_pretrained_PLORASPRETRAIN --kwargs sim_loss_weight=25 var_loss_weight=25 cov_loss_weight=0.25`

#### Hypernet
`hypunet_train 3d_fullres VICRegTrainer 002 all -n hyper-0-s25-v25-c0.25 -p hypunetPlans_pretrained_PLORASPRETRAIN --meta 4 --hyper 0 --kwargs sim_loss_weight=25 var_loss_weight=25 cov_loss_weight=0.25`

### GradCache

#### Baseline
`hypunet_train 3d_fullres GC_VICRegTrainer 002 all -n baseline-s25-v25-c0.25 -p hypunetPlans_pretrained_PLORASPRETRAIN --kwargs metabatch=16 sim_loss_weight=25 var_loss_weight=25 cov_loss_weight=0.25`

#### Hypernet
`hypunet_train 3d_fullres GC_VICRegTrainer 002 all -n hyper-0-s25-v25-c0.25 -p hypunetPlans_pretrained_PLORASPRETRAIN --meta 4 --hyper 0 --kwargs metabatch=16 sim_loss_weight=25 var_loss_weight=25 cov_loss_weight=0.25`

## Training

### Standard

#### Baseline
`hypunet_train 3d_fullres HyperTrainerV2 001 0 -n baseline`

#### Hypernet
`hypunet_train 3d_fullres HyperTrainerV2 001 0 -n hyper-0 --meta 4 --hyper 0`

### Pre-trained

#### Baseline
`hypunet_train 3d_fullres HyperTrainerV2 001 0 -n baseline-s25-c25-v0.25 -p hypunetPlans_pretrained_PLORASPRETRAIN -pretrained_weights ~/hypunet/hypunet_trained_models/hypunet/3d_fullres/Task002_CLINICALPLORAS/GC_VICRegTrainer__hypunetPlans_pretrained_PLORASPRETRAIN/baseline-s25-c25-v0.25/model_final_checkpoint.model`

#### Hypernet
`hypunet_train 3d_fullres HyperTrainerV2 001 0 -n hyper-0-s25-c25-v0.25 --meta 4 --hyper 0 -p hypunetPlans_pretrained_PLORASPRETRAIN -pretrained_weights ~/hypunet/hypunet_trained_models/hypunet/3d_fullres/Task002_CLINICALPLORAS/GC_VICRegTrainer__hypunetPlans_pretrained_PLORASPRETRAIN/hyper-0-s25-c25-v0.25/model_final_checkpoint.model`
