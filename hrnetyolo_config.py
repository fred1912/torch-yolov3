from easydict import EasyDict
import os.path as osp


OPTIM=EasyDict()
OPTIM.WARMUP_EPOCH=2
OPTIM.EPOCH=30
OPTIM.INIT_LR=1e-4
OPTIM.END_LR=1e-6

TRAIN=EasyDict()
TRAIN.DEVICE='cuda'
TRAIN.LOGS=osp.join(osp.dirname(__file__),'logs','hrnet_yolo')
TRAIN.CHECKPOINT=osp.join(osp.dirname(__file__),'weights','hrnet_yolo')

DATASET=EasyDict()
DATASET.data_dir='/data/yoloCao/DataSet/person_helmet/coco2017'
DATASET.keep_res=False
DATASET.pad=0
DATASET.input_h=416
DATASET.input_w=416
DATASET.not_rand_crop=False
DATASET.flip=0.5
DATASET.no_color_aug=False
DATASET.mse_loss=False
DATASET.dense_wh=False
DATASET.cat_spec_wh=False
DATASET.reg_offset=True
DATASET.debug=0
DATASET.DOWN_RATION=[8,16,32]
DATASET.BATCHSIZE=14
DATASET.ANCHOR=[
                [(3.625, 2.8125), (4.875, 6.1875), (11.65625, 10.1875)],# Anchors for big obj
                [(1.875, 3.8125), (3.875, 2.8125), (3.6875, 7.4375)],  # Anchors for medium obj
                [(1.25, 1.625), (2.0, 3.75), (4.125, 2.875)],       # Anchors for small obj
                ]


MODEL=EasyDict()
MODEL.BACKBONE='hrnet'
MODEL.INIT_WEIGHTS = True
MODEL.PRETRAINED = osp.join(osp.dirname(__file__),
                            'weights/pretrained/hrnet_w32-36af842e.pth')
MODEL.PRETRAINED_LAYERS = ['*']
MODEL.HEADS={'yolo-0':3*(80+5),
             'yolo-1':3*(80+5),
             'yolo-2':3*(80+5)}

MODEL.STAGE2=EasyDict()
MODEL.STAGE2.NUM_MODULES = 1
MODEL.STAGE2.NUM_BRANCHES = 2
MODEL.STAGE2.NUM_BLOCKS = [4, 4]
MODEL.STAGE2.NUM_CHANNELS = [32, 64]
MODEL.STAGE2.BLOCK = 'BASIC'
MODEL.STAGE2.FUSE_METHOD = 'SUM'

MODEL.STAGE3 = EasyDict()
MODEL.STAGE3.NUM_MODULES = 4
MODEL.STAGE3.NUM_BRANCHES = 3
MODEL.STAGE3.NUM_BLOCKS = [4, 4, 4]
MODEL.STAGE3.NUM_CHANNELS = [32, 64, 128]
MODEL.STAGE3.BLOCK = 'BASIC'
MODEL.STAGE3.FUSE_METHOD = 'SUM'

MODEL.STAGE4 = EasyDict()
MODEL.STAGE4.NUM_MODULES = 3
MODEL.STAGE4.NUM_BRANCHES = 4
MODEL.STAGE4.NUM_BLOCKS = [4, 4, 4, 4]
MODEL.STAGE4.NUM_CHANNELS = [32, 64, 128, 256]
MODEL.STAGE4.BLOCK = 'BASIC'
MODEL.STAGE4.FUSE_METHOD = 'SUM'





