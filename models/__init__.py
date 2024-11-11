# -*-coding:utf-8-*-
from .oenet import *
from .alexnet import *


def get_model(config):
    if 'params' in config.architecture:
        model = globals()[config.architecture.type](config.num_classes, **config.architecture.params)
    else:
        model = globals()[config.architecture.type](config.num_classes)
    return model
