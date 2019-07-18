from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import pycocotools.coco as coco
import numpy as np
import os

import torch.utils.data as data


class PascalVOC(data.Dataset):
    def __init__(self, config, split):
        super(PascalVOC, self).__init__()
        config=config.DATASET
        self.data_dir = config.DATA_DIR
        self.img_dir = os.path.join(self.data_dir, 'images')
        _ann_name = {'train': 'trainval0712', 'val': 'test2007'}
        self.annot_path = os.path.join(
            self.data_dir, 'annotations',
            'pascal_{}.json').format(_ann_name[split])
        self.max_objs = 80
        self.class_name = ['__background__', "aeroplane", "bicycle", "bird", "boat",
                           "bottle", "bus", "car", "cat", "chair", "cow", "diningtable", "dog",
                           "horse", "motorbike", "person", "pottedplant", "sheep", "sofa",
                           "train", "tvmonitor"]
        self._valid_ids = np.arange(1, 21, dtype=np.int32)
        self.cat_ids = {v: i for i, v in enumerate(self._valid_ids)}
        self._data_rng = np.random.RandomState(123)
        self.num_classes = 20

        self.mean = np.array([0.406, 0.456, 0.485 ],
                        dtype=np.float32).reshape(1, 1, 3)
        self.std = np.array([ 0.225, 0.224,0.229],
                       dtype=np.float32).reshape(1, 1, 3)

        self.split = split
        self.config = config

        print('==> initializing pascal {} data.'.format(_ann_name[split]))
        self.coco = coco.COCO(self.annot_path)
        self.images = sorted(self.coco.getImgIds())
        self.num_samples = len(self.images)

        print('Loaded {} {} samples'.format(split, self.num_samples))

    def _to_float(self, x):
        return float("{:.2f}".format(x))

    def __len__(self):
        return self.num_samples

    def _get_border(self, border, size):
        i = 1
        while size - border // i <= border // i:
            i *= 2
        return border // i

    def _coco_box_to_bbox(self, box):
        bbox = np.array([box[0], box[1], box[0] + box[2], box[1] + box[3]],
                        dtype=np.float32)
        return bbox
    
    def __getitem__(self, index):
        raise NotImplementedError

