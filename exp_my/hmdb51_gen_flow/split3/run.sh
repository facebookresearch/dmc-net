#! /bin/bash

expdir=hmdb51_gen_flow/split3
representation=mv

# exp/hmdb51_gen_flow/split3/run.sh; ./exp/hmdb51_gen_flow/split3/run_combine.sh 2>&1 | tee ./exp/hmdb51_gen_flow/split3/acc.log

python train.py \
	--lr 0.01 \
	--batch-size 45 \
	--arch resnet18 \
	--arch_estimator DenseNetTiny \
	--data-name hmdb51 \
	--representation ${representation} \
	--data-root /projects/eventnet/dataset/HMDB51/fb/videos_mpeg4 \
 	--flow-root /projects/eventnet/dataset/HMDB51/fb/TSN_input \
	--train-list /projects/LSDE/work03/FB/data_preprocess/datalists/hmdb51_split3_train_rename.txt \
	--test-list /projects/LSDE/work03/FB/data_preprocess/datalists/hmdb51_split3_test_rename.txt \
	--weights ./exp/hmdb51_coviar/flow/split3/_flow_model_best.pth.tar \
	--model-prefix exp/${expdir}/ \
	--lr-steps 20 35 45 \
	--lr-mse 10 \
	--lr_mse_mult 1 \
	--use_databn 0 \
	--epochs 50 \
	--epoch-thre 1 \
	--flow_ds_factor 16 \
	--gen_flow_or_delta 1 \
	--no-accumulation \
	--mv_minmaxnorm 1 \
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
	--flow_ds_factor 16 \
	--gen_flow_or_delta 1 \
	--no-accumulation \
	--mv_minmaxnorm 1 \
	--save-scores exp/${expdir}/${representation}_score_model_best \
	--gpus 0 2>&1 | tee exp/${expdir}/test.log