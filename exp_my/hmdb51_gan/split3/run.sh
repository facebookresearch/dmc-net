# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

#! /bin/bash

expdir=hmdb51_gan/split3
representation=mv

# exp/hmdb51_gan/split3/run.sh; ./exp/hmdb51_gan/split3/run_combine.sh 2>&1 | tee ./exp/hmdb51_gan/split3/acc.log

python train.py \
	--lr 0.01 \
	--lr-adv-g 1 \
	--lr-adv-d 0.01 \
	--lr-mse 10 \
	--lr_mse_mult 1 \
	--lr_d_mult 1 \
	--batch-size 30 \
	--arch resnet18 \
	--arch_estimator DenseNetTiny \
	--arch_d Discriminator3 \
	--data-name hmdb51 \
	--representation ${representation} \
 	--data-root /projects/eventnet/dataset/HMDB51/fb/videos_mpeg4 \
 	--flow-root /projects/eventnet/dataset/HMDB51/fb/TSN_input \
 	--train-list /projects/LSDE/work03/FB/data_preprocess/datalists/hmdb51_split3_train_rename.txt \
 	--test-list /projects/LSDE/work03/FB/data_preprocess/datalists/hmdb51_split3_test_rename.txt \
	--weights exp/hmdb51_gen_flow/split3/_mv_model_best.pth.tar \
	--model-prefix exp/${expdir}/ \
	--lr-steps 20 35 45 \
	--use_databn 0 \
	--epochs 50 \
	--epoch-thre 0 \
	--flow_ds_factor 0 \
	--gen_flow_or_delta 1 \
	--mv_minmaxnorm 1 \
	--no-accumulation \
	--gpus 0 2>&1 | tee exp/${expdir}/train.log

python test.py \
	--arch resnet18 \
	--arch_estimator DenseNetTiny \
	--data-name hmdb51 \
	--representation mv \
	--test-crops 1 \
	--test_segments 25 \
 	--data-root /projects/eventnet/dataset/HMDB51/fb/videos_mpeg4 \
 	--flow-root /projects/eventnet/dataset/HMDB51/fb/TSN_input \
  	--test-list /projects/LSDE/work03/FB/data_preprocess/datalists/hmdb51_split3_test_rename.txt \
	--weights exp/${expdir}/_${representation}_model_best.pth.tar \
	--use_databn 0 \
	--flow_ds_factor 0 \
	--gen_flow_or_delta 1 \
	--mv_minmaxnorm 1 \
	--no-accumulation \
	--save-scores exp/${expdir}/${representation}_score_model_best \
	--gpus 0 2>&1 | tee exp/${expdir}/test.log