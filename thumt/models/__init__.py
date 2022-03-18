# coding=utf-8
# Copyright 2017-2020 The THUMT Authors

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import thumt.models.msp
import thumt.models.prompt
import thumt.models.prefix


def get_model(name):
    name = name.lower()

    if name == "mgpt_prefix":
        return thumt.models.prefix.mGPT
    elif name == "mgpt_prompt":
        return thumt.models.prompt.mGPT
    elif name == "mgpt_msp":
        return thumt.models.msp.mGPT
    else:
        raise LookupError("Unknown model %s" % name)
