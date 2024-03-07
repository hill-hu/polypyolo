from multiprocessing import freeze_support

import logging
import os
import argparse

import sys
from datetime import timedelta

from ultralytics import YOLO, utils


def parse_opt(known=False):
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', type=str, default='', help='model.yaml path')
    parser.add_argument("--data", type=str, help="data.yaml path ")

    opt = parser.parse_known_args()[0] if known else parser.parse_args()
    return opt


if __name__ == '__main__':
    freeze_support()
    # Load a model
    opts = parse_opt()
    print(opts)
    cfg = utils.yaml_load(opts.cfg)

    model = YOLO(cfg['pretrained'])  # load a pretrained model (recommended for training)
    # model = YOLO('yolov8n.yaml').load('yolov8n.pt')  # build from YAML and transfer weights

    # Train the model
    results = model.train(cfg=opts.cfg, data=opts.data)
