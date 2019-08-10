# DMC-Net

### Citing
If you find DMC-Net useful, please consider citing:

    @inproceedings{shou2019dmc,
    	title={DMC-Net: Generating Discriminative Motion Cues for Fast Compressed Video Action Recognition},
    	author={Shou, Zheng and Lin, Xudong and Kalantidis, Yannis and Sevilla-Lara, Laura and Rohrbach, Marcus and Chang, Shih-Fu and Yan, Zhicheng},
    	booktitle={Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR)},
    	year={2019}
    }

### Overview

`./exp_my/` contains scripts for running experiments and our trained models and prediction results.

`./code/` contains implementation for 3 major models respectively: 
0. `dmcnet` indicates the version which does not include the adversarial loss during training and uses ResNet-18 for classifying DMC;
1. `dmcnet_GAN` indicates the version which includes the adversarial loss during training and uses ResNet-18 for classifying DMC;
2. `dmcnet_I3D` indicates the version which uses I3D for classifying DMC.

**In the following, we present how to use `dmcnet` and `dmcnet_GAN`. Instructions for `dmcnet_I3D` can be found in `./code/dmcnet_I3D/`.**

## DMC-Net with ResNet-18 classifier

### Installation

We implement `dmcnet` and `dmcnet_GAN` using PyTorch based on [CoViAR](https://github.com/chaoyuaw/pytorch-coviar). Please refer to CoViAR for details of setup and installation (e.g. how to prepare input videos, setup mpeg-4 compressed video data loader, etc.). Specifically, the released models were trained using python 3.6, pytorch 0.31, cuda 9.0, MPEG-4 video of GOP 12 and macroblock size 16x16.

Optical flow extraction: we extract optical flow using TV-L1 algorithm implementation from [dense_flow](https://github.com/wanglimin/dense_flow) and store the flow images beforehand and then load flow images during training.

In both `./code/dmcnet/` and `./code/dmcnet_GAN/`, please first link `exp/` to true directory of './exp_my/' so that all data will be stored in the experimental folder. 

### Usage

As stated in the paper, we first train DMC-Net with the classification loss and flow reconstruction MSE loss but without the adversarial loss (using `./code/dmcnet/`). Sample training script for HMDB-51 can be found at `exp_my/hmdb51_gen_flow/split1/run.sh`. Performing training and testing by `exp/hmdb51_gen_flow/split1/run.sh;`. The trained model would be `exp/hmdb51_gen_flow/split1/_mv_model_best.pth.tar`.

Explanations about some key options used in the `run.sh` script (detaied descriptions can be found in `train_options.py`):

0. `data-root`: specify the directory for storing mpeg-4 videos;
1. `train-list` and `test-list`: specify the training and testing videos lists. Some example lines in such list files (format follows [CoViAR](https://github.com/chaoyuaw/pytorch-coviar): directory class class_index): 

    smile/Me_smiling_smile_h_nm_np1_fr_goo_0.avi smile 0
    
    clap/Alex_applauding_himself_clap_u_nm_np1_fr_med_0.avi clap 1
    
    climb/Chiara_Kletterwand_climb_f_cm_np1_ba_bad_0.avi climb 2

2. `flow-root`: specify the directory for storing ground truth optical flow images extracted by [dense_flow](https://github.com/wanglimin/dense_flow). Sample directory:

    flow-root/brush_hair/April_09_brush_hair_u_nm_np1_ba_goo_0/flow_x_00001.jpg
    
    flow-root/brush_hair/April_09_brush_hair_u_nm_np1_ba_goo_0/flow_y_00001.jpg
    
    flow-root/brush_hair/April_09_brush_hair_u_nm_np1_ba_goo_0/flow_x_00002.jpg
    
    flow-root/brush_hair/April_09_brush_hair_u_nm_np1_ba_goo_1/flow_x_00001.jpg
    
    flow-root/climb_stairs/BIG_FISH_climb_stairs_f_nm_np1_fr_med_1/flow_x_00001.jpg

Then we use the above trained model as initialization to train with the adversarial loss included (using `./code/dmcnet_GAN/`). Sample training script for HMDB-51 can be found at `exp_my/hmdb51_gan/split1/run.sh`. In order to reproduce the result on HMDB-51, simply run: `bash exp/hmdb51_gan/split1/run.sh; bash ./exp/hmdb51_gan/split1/run_combine.sh 2>&1 | tee ./exp/hmdb51_gan/split1/acc.log` The trained model would be `exp/hmdb51_gan/split1/_mv_model_best.pth.tar` and the prediction results would be stored in `exp/hmdb51_gan/split1/mv_score_model_best.npz` and `./exp/hmdb51_gan/split1/acc.log` records the accuracy after fusing all modalities.

### Our trained models

At AWS host [here](dl.fbaipublicfiles.com/dmc-net/models.zip), we provide our trained models and prediction results. The file directory of `./models/` follows similar structure as `./exp_my/`. Please put the trained model and prediction result (for each dataset and split from `./models/`) in the corresponding folder (for experiment in `./exp_my/`). 

### Results

Accuracy (%)     | HMDB-51 | UCF-101
---------|--------|-----
[EMV-CNN](https://ieeexplore.ieee.org/abstract/document/7780666)     | 51.2 (split1) | 86.4
[DTMV-CNN](https://zbwglory.github.io/papers/08249882.pdf)     | 55.3 | 87.5
[CoViAR](https://github.com/chaoyuaw/pytorch-coviar)     | 59.1 | 90.4
DMC-Net (ResNet-18)     | 62.8 | 90.9
DMC-Net (I3D)     | 71.8 | 92.3
DMC-Net (I3D) + I3D RGB     | 77.8 | 96.5

## License
DMC-Net is MIT licensed, as found in the LICENSE file.