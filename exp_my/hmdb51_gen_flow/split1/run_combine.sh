#! /bin/bash

# ./exp/hmdb51_gen_flow/split1/run_combine.sh 2>&1 | tee ./exp/hmdb51_gen_flow/split1/acc.log

expdir=hmdb51_gen_flow/split1
representation=mv

python combine.py \
  --iframe exp/hmdb51_coviar/iframe/split1/iframe_score_model_best.npz \
  --res exp/hmdb51_coviar/residual/split1/residual_score_model_best.npz \
  --mv exp/hmdb51_coviar/mv/split1/mv_score_model_best.npz \
  --flow exp/${expdir}/mv_score_model_best.npz