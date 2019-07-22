# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# ./exp/hmdb51_coviar/flow/split2/combine.sh 2>&1 | tee ./exp/hmdb51_coviar/flow/split2/acc.log

expdir=hmdb51_coviar
expname=flow/split2

python combine.py --iframe exp/hmdb51_coviar/iframe/split2/iframe_score_model_best.npz --res exp/hmdb51_coviar/residual/split2/residual_score_model_best.npz \
  --mv exp/hmdb51_coviar/mv/split2/mv_score_model_best.npz --flow exp/${expdir}/${expname}/flow_score_model_best.npz
