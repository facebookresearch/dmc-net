# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

python evaluate_video_hmdb_i3d.py --task-name hmdb1\
                                    --split 1\
                                    --load-epoch 10\
                                    --modality flow+mp4\
                                    --log-file ./eval_hmdb1.log \
                                    --gpus 0,1\
                                    --batch-size 2  --clip-length 250\
                                    --arch-estimator DenseNetTiny\
                                    --accumulate 0\
                                    --mv-minmaxnorm 1 

