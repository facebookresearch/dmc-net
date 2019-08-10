# DMC-Net + I3D


This directory contains the code used for the experiment of combining our DMC-Net and I3D.


## Dependencies
Our code is built on the the following (but not limited to) packages:
1. PyTorch 0.4.0
2. Python 3.6, numpy
3. [coviar](https://github.com/chaoyuaw/pytorch-coviar/blob/master/GETTING_STARTED.md)

## Data
The experiments are done based on mpeg4 videos of [HMDB51](http://serre-lab.clps.brown.edu/resource/hmdb-a-large-human-motion-database/) and [UCF101](https://www.crcv.ucf.edu/data/UCF101.php). 

Path to videos or stored MV and R is supposed to be manually set in data/video_iterator.py.

## Training
First please go to [here](https://github.com/hassony2/kinetics_i3d_pytorch) to download the pretrained flow model of I3D.
We have two separate files for HMDB51 and UCF101. We first train the generator without updating the weights of classifier (I3D) using Reconstruction loss and advasarial loss and classification loss. We train it in this way for epoch thre epochs. Then we joint update generator and classifier. Discriminator is updated in both stages.
The details of how to use them are shown in the following sample training script.
- Sample script for training on HMDB51 split 1
  ```
  bash train.sh
  ```
## Testing
Open the directory test. We have two separate files for HMDB51 and UCF101. The details of how to use them are shown in the following sample testing code. Please put the model you want to evaluate in ./exps/models/.
- Sample script for testing on HMDB51 split 1
  ```
  bash test.sh
  ```
We provide models that produce the results reported in our paper [here]().

## Acknowledgment
Our training and testing code is mainly built on [MF-Net](https://github.com/cypw/PyTorch-MFNet). Our I3D model is borrowed from  this pytorch [implementation](https://github.com/hassony2/kinetics_i3d_pytorch). Our dataloader also borrows code from [CoViAR](https://github.com/chaoyuaw/pytorch-coviar). Thanks a lot!





