from easydict import EasyDict
from collections import OrderedDict
import os.path as osp
dataset={
        'coco':
            {'path':'/data/DataSet/person_helmet/coco2017','num_class':80},
        'pascal':
            {'path':'/data/DataSet/VOC','num_class':20},
        'objects365':'',
        }

DATASET=EasyDict()
DATASET.NAME = 'pascal'
DATASET.DATA_DIR=dataset[DATASET.NAME]['path']
DATASET.BATCHSIZE=10
DATASET.DOWN_RATION=[8,16,32]
DATASET.INPUT_H=416
DATASET.INPUT_W=416
DATASET.USE_GIOU=False
DATASET.ANCHORS=[[[0.01402, 0.02311], [0.02464, 0.06542], [0.05019, 0.13322]],
                 [[0.05088, 0.03728], [0.10116, 0.08054], [0.10545, 0.26739]],
                 [[0.21067, 0.15731], [0.27648, 0.42582], [0.69417, 0.71236]]]

BACKBONE=EasyDict()
BACKBONE.INIT_WEIGHTS=True
BACKBONE.PRETRAINED=osp.join(osp.dirname(__file__),'..','weights/pretrained/darknet53_74.pth')

NECK=EasyDict()
NECK.IN_CHANNEL=[256,512,1024]

HEAD=EasyDict()
HEAD.BACKBONE='darknet'
HEAD.IN_CHANNEL=[128,256,512]
HEAD.OUT_HEADS={
                'yolo-0':len(DATASET.ANCHORS[0])*(dataset[DATASET.NAME]['num_class']+5),
                'yolo-1':len(DATASET.ANCHORS[0])*(dataset[DATASET.NAME]['num_class']+5),
                'yolo-2':len(DATASET.ANCHORS[0])*(dataset[DATASET.NAME]['num_class']+5)
                }

OPTIM=EasyDict()
OPTIM.WARMUP_EPOCH=2
OPTIM.EPOCH=1000
OPTIM.INIT_LR=8e-4
OPTIM.END_LR=1e-6
OPTIM.WEIGHT_DECAY=5e-4

TRAIN=EasyDict()
TRAIN.DEVICE='cuda'
TRAIN.LOGS=osp.join(osp.dirname(__file__),'..','logs','dark_yolo')
TRAIN.CHECKPOINT=osp.join(osp.dirname(__file__),'..','weights','dark_yolo')