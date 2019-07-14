from easydict import EasyDict
import os.path as osp


OPTIM=EasyDict()
OPTIM.WARMUP_EPOCH=2
OPTIM.EPOCH=30
OPTIM.INIT_LR=1e-4
OPTIM.END_LR=1e-6

TRAIN=EasyDict()
TRAIN.DEVICE='cuda'
TRAIN.LOGS=osp.join(osp.dirname(__file__),'logs','dark_yolo')
TRAIN.CHECKPOINT=osp.join(osp.dirname(__file__),'weights','dark_yolo')

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
DATASET.BATCHSIZE=16
DATASET.ANCHOR=[
                [(3.625, 2.8125), (4.875, 6.1875), (11.65625, 10.1875)],# Anchors for big obj
                [(1.875, 3.8125), (3.875, 2.8125), (3.6875, 7.4375)],  # Anchors for medium obj
                [(1.25, 1.625), (2.0, 3.75), (4.125, 2.875)],       # Anchors for small obj
                ]
MODEL=EasyDict()
MODEL.BACKBONE='darknet'
MODEL.INIT_WEIGHTS = True
MODEL.PRETRAINED = osp.join(osp.dirname(__file__),
                            'weights/pretrained/darknet53.pth')
MODEL.PRETRAINED_LAYERS = ['*']
MODEL.HEADS={'yolo-0':3*(80+5),
             'yolo-1':3*(80+5),
             'yolo-2':3*(80+5)}

MODEL.STAGE4 = EasyDict()
MODEL.STAGE4.NUM_CHANNELS = [128, 256, 512, 1024]





