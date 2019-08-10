"""
Copyright (c) Facebook, Inc. and its affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

import logging

def get_config(name, modality = 'rgb', **kwargs):

    logging.debug("loading network configs of: {}".format(name.upper()))

    config = {}

    if name.upper() == "I3D":
        config['mean'] = [0.5] * 3
        config['std'] = [0.5] * 3
    else:
        config['mean'] = [0.485, 0.456, 0.406]
        config['std'] = [0.229, 0.224, 0.225]
    
    # else:
    #    raise NotImplemented("Configs for {} not implemented".format(name))

    logging.info("data:: {}".format(config))
    return config