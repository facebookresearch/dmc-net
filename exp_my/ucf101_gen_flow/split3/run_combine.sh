#! /bin/bash

# ./exp/ucf101_gen_flow/split3/run_combine.sh 2>&1 | tee ./exp/ucf101_gen_flow/split3/acc.log

expdir=ucf101_gen_flow/split3
representation=mv

python combine.py \
  --iframe exp/ucf101_coviar/ucf101_iframe/split3/iframe_score_model_best.npz \
  --res exp/ucf101_coviar/ucf101_residual/split3/residual_score_model_best.npz \
  --mv exp/ucf101_coviar/ucf101_mv/split3/mv_score_model_best.npz \
  --flow exp/${expdir}/mv_score_model_best.npz \
  --wf 0.25