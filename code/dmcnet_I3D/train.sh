# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

python train_hmdb51.py --task-name hmdb_1\
                       --split 1\
                       --network I3D \
                       --clip-length 64 \
                       --pretrained_3d ./exps/models/model_flow.pth\
                       --iter-size 32 --batch-size 3\
                       --optimizer adam\
                       --gpus 1,3\
                       --modality flow+mp4\
                       --train-frame-interval 1 \
                       --val-frame-interval 1\
                       --lr-base 0.0004\
                       --lr-base2 0.0004\
                       --lr-d 0.002\
                       --detach 1\
                       --lr-factor 0.2\
                       --dataset HMDB51\
                       --drop-out 0.85\
                       --fine_tune 0\
                       --arch-estimator DenseNetTiny\
                       --arch-d Discriminator\
                       --adv 1\
                       --epoch-thre 6\
                       --ds_factor 16\
                       --mv-minmaxnorm 1\
                       --accumulate 0\