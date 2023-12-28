#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Copyright (c) Megvii, Inc. and its affiliates.

import os

import torch.nn as nn

from yolox.exp import Exp as MyExp

# available img sizes
# 96 128
# 192 256
# 288 384
# 384 512

SIZE = (192, 256)
# SIZE = (160, 160)


class Exp(MyExp):
    def __init__(self):
        super(Exp, self).__init__()
        self.depth = 0.33
        # self.width = 0.25
        self.width = 0.125
        self.input_size = SIZE
        self.mosaic_scale = (0.75, 1.25)
        self.mosaic_prob = 0
        # self.random_size = (10, 20)
        self.test_size = SIZE
        # self.exp_name = os.path.split(os.path.realpath(__file__))[1].split(".")[0]
        self.exp_name = "pg_coco_rgb_192_no_mosaic_w_pretrained"
        # self.exp_name = "pg_coco_160_full_exp_3"
        self.enable_mixup = False

        # Define yourself dataset path
        self.data_dir = "datasets/pg_coco"
        self.train_ann = "instances_train.json"
        self.val_ann = "instances_val.json"
        self.use_rgb = False

        self.train_name = "train2017"
        self.val_name = "val2017"
        self.test_name = "test2017"
        # self.data_dir = "datasets/coco128"
        # self.train_ann = "instances_train2017.json"
        # self.val_ann = "instances_val2017.json"

        self.num_classes = 1
        self.max_epoch = 1000

    def get_model(self, sublinear=False):

        def init_yolo(M):
            for m in M.modules():
                if isinstance(m, nn.BatchNorm2d):
                    m.eps = 1e-3
                    m.momentum = 0.03
        if "model" not in self.__dict__:
            from yolox.models import YOLOX, YOLOPAFPN, YOLOXHead
            in_channels = [256, 512, 1024]
            # NANO model use depthwise = True, which is main difference.
            backbone = YOLOPAFPN(self.depth, self.width, in_channels=in_channels, depthwise=True)
            head = YOLOXHead(self.num_classes, self.width, in_channels=in_channels, depthwise=True)
            self.model = YOLOX(backbone, head)

        self.model.apply(init_yolo)
        self.model.head.initialize_biases(1e-2)
        return self.model
