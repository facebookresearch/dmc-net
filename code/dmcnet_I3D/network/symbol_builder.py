"""
Copyright (c) Facebook, Inc. and its affiliates.
This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

import logging

from .i3d import I3D
from .config import get_config

def get_symbol(name, modality = 'rgb', drop_out = 0.5, print_net=False, arch_estimator = None, arch_d = None, **kwargs):

    if name.upper() == "I3D":
        net = I3D(modality = modality, dropout_prob = drop_out, arch_estimator = arch_estimator, arch_d = arch_d, **kwargs)
    else:
        logging.error("network '{}'' not implemented".format(name))
        raise NotImplementedError()

    if print_net:
        logging.debug("Symbol:: Network Architecture:")
        logging.debug(net)

    input_conf = get_config(name, modality = modality, **kwargs)
    return net, input_conf
