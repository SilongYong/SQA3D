# ---------------------------------------------------------------
# Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.
#
# This file has been modified from a file in the Bongard-HOI library
# which was released under the NVIDIA Source Code Licence.
#
# Source:
# https://github.com/NVlabs/Bongard-HOI
#
# The license for the original version of this file can be
# found in https://github.com/NVlabs/Bongard-HOI/blob/master/LICENSE
# The modifications to this file are subject to the same NVIDIA Source Code Licence.
# ---------------------------------------------------------------

import torch


models = {}
def register(name):
    def decorator(cls):
        models[name] = cls
        return cls
    return decorator


def make(name, **kwargs):
    if name is None:
        return None
    model = models[name](**kwargs)
    # if torch.cuda.is_available():
    #     model = model.cuda()
    return model


def load(model_sv, name=None):
    if name is None:
        name = 'model'
    model = make(model_sv[name], **model_sv[name + '_args'])
    missing_keys, unexpected_keys = model.load_state_dict(model_sv[name + '_sd'])
    return model
