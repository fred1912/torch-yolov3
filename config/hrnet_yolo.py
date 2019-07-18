from easydict import EasyDict
import os.path as osp
import numpy as np

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
DATASET.BATCHSIZE=16
DATASET.DOWN_RATION=[4,8,16,32]
DATASET.INPUT_H=416
DATASET.INPUT_W=416
DATASET.USE_GIOU=False
DATASET.ANCHORS=[[[0.01181, 0.01955], [0.01941, 0.05309], [0.03584, 0.1]],
                 [[0.03831, 0.02876], [0.06861, 0.05756], [0.06994, 0.16855]],
                 [[0.11966, 0.3311], [0.13606, 0.0961], [0.21739, 0.1955]],
                 [[0.25938, 0.5573], [0.47191, 0.32135], [0.72506, 0.77527]]]

BACKBONE=EasyDict()
BACKBONE.INIT_WEIGHTS = True
BACKBONE.PRETRAINED = osp.join(osp.dirname(__file__),'..',
                            'weights/pretrained/hrnet_w32-36af842e.pth')
BACKBONE.PRETRAINED_LAYERS = ['*']

BACKBONE.STAGE2=EasyDict()
BACKBONE.STAGE2.NUM_MODULES = 1
BACKBONE.STAGE2.NUM_BRANCHES = 2
BACKBONE.STAGE2.NUM_BLOCKS = [4, 4]
BACKBONE.STAGE2.NUM_CHANNELS = [32, 64]
BACKBONE.STAGE2.BLOCK = 'BASIC'
BACKBONE.STAGE2.FUSE_METHOD = 'SUM'

BACKBONE.STAGE3 = EasyDict()
BACKBONE.STAGE3.NUM_MODULES = 4
BACKBONE.STAGE3.NUM_BRANCHES = 3
BACKBONE.STAGE3.NUM_BLOCKS = [4, 4, 4]
BACKBONE.STAGE3.NUM_CHANNELS = [32, 64, 128]
BACKBONE.STAGE3.BLOCK = 'BASIC'
BACKBONE.STAGE3.FUSE_METHOD = 'SUM'

BACKBONE.STAGE4 = EasyDict()
BACKBONE.STAGE4.NUM_MODULES = 3
BACKBONE.STAGE4.NUM_BRANCHES = 4
BACKBONE.STAGE4.NUM_BLOCKS = [4, 4, 4, 4]
BACKBONE.STAGE4.NUM_CHANNELS = [32, 64, 128, 256]
BACKBONE.STAGE4.BLOCK = 'BASIC'
BACKBONE.STAGE4.FUSE_METHOD = 'SUM'


NECK=EasyDict()
NECK.IN_CHANNEL= [32, 64, 128, 256]

HEAD=EasyDict()
HEAD.BACKBONE='hrnet'
HEAD.IN_CHANNEL=[32, 64, 128, 256]
HEAD.OUT_HEADS={
                'yolo-0':len(DATASET.ANCHORS[0])*(dataset[DATASET.NAME]['num_class']+5),
                'yolo-1':len(DATASET.ANCHORS[0])*(dataset[DATASET.NAME]['num_class']+5),
                'yolo-2':len(DATASET.ANCHORS[0])*(dataset[DATASET.NAME]['num_class']+5),
                'yolo-3':len(DATASET.ANCHORS[0])*(dataset[DATASET.NAME]['num_class']+5)
                }

OPTIM=EasyDict()
OPTIM.WARMUP_EPOCH=2
OPTIM.EPOCH=1000
OPTIM.INIT_LR=8e-4
OPTIM.END_LR=1e-6
OPTIM.WEIGHT_DECAY=5e-4

TRAIN=EasyDict()
TRAIN.DEVICE='cuda'
TRAIN.LOGS=osp.join(osp.dirname(__file__),'..','logs','hrnet_yolo')
TRAIN.CHECKPOINT=osp.join(osp.dirname(__file__),'..','weights','hrnet_yolo')
