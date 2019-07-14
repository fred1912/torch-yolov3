import os
os.environ.setdefault('CUDA_VISIBLE_DEVICES','0')

from dataset.yolo_coco import yoloCOCO
from models.yolo_head import Yolodet
import darkyolo_config as cfg
from torch.optim import Adam
from torch.utils.data import DataLoader
from utils.trainer import YolodetTrainer

model = Yolodet(cfg)
train_loader = DataLoader(yoloCOCO(cfg, 'train'), batch_size=cfg.DATASET.BATCHSIZE, shuffle=True, num_workers=4)
val_loader = DataLoader(yoloCOCO(cfg, 'val'), batch_size=cfg.DATASET.BATCHSIZE, shuffle=True, num_workers=4)
opti = Adam(model.parameters(), lr=cfg.OPTIM['INIT_LR'])
trainer = YolodetTrainer(cfg, model, opti, [train_loader, val_loader])
trainer.set_device('cuda')
trainer.run()