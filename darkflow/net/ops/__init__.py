#!/usr/bin/env python
# -*- coding: utf-8 -*-

from .simple import *
from .convolution import *
from .baseop import HEADER, LINE


###
#  List of Operation Types. These operations are tensorflow-compatible
#  implementations of the network layers.
#
#   Extend this list to create new operations.
###
op_types = {
    'convolutional': convolutional,
    'conv-select': conv_select,
    'connected': connected,
    'maxpool': maxpool,
    'leaky': leaky,
    'dropout': dropout,
    'flatten': flatten,
    'avgpool': avgpool,
    'softmax': softmax,
    'identity': identity,
    'crop': crop,
    'local': local,
    'select': select,
    'route': route,
    'reorg': reorg,
    'conv-extract': conv_extract,
    'extract': extract
}


def op_create(*args):
    """ TODO Documentation"""
    layer_type = list(args)[0].type
    return op_types[layer_type](*args)