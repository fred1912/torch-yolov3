from dataset.ctdet_coco import ctdetCOCO
from models.ct_head import Ctdet
import ctdet_config as cfg
from torch.optim import SGD
from torch.utils.data import DataLoader
from utils.trainer import CTdetTrainer
model = Ctdet(cfg)
train_loader = DataLoader(ctdetCOCO(cfg, 'train'), batch_size=cfg.DATASET.BATCHSIZE, shuffle=True, num_workers=4)
val_loader = DataLoader(ctdetCOCO(cfg, 'val'), batch_size=cfg.DATASET.BATCHSIZE, shuffle=True, num_workers=4)
opti = SGD(model.parameters(), lr=cfg.OPTIM['INIT_LR'])
trainer = CTdetTrainer(cfg, model, opti, [train_loader, val_loader])
trainer.set_device('cuda')
trainer.run()