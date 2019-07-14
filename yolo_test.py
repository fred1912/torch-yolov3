from dataset.yolo_coco import yoloCOCO
from models.yolo_head import Yolodet
import hrnetyolo_config as hrnet_cfg
import darkyolo_config as dark_cfg
from torch.utils.data import DataLoader
from utils.checkpoint import load_checkpoint
from utils.decoder import convert_pred
import torch
import cv2
import numpy as np

cfg=hrnet_cfg
weights='weights/hrnet_yolo/model_0.pth'

model = Yolodet(cfg,pretrained=False)
load_checkpoint(model,weights)
model.eval()
val_loader = DataLoader(yoloCOCO(cfg, 'val'), batch_size=1, shuffle=True, num_workers=1)
for batch in val_loader:
    out=model(batch['input'])
    img=batch['image'][0].numpy()
    h,w,_=img.shape
    result=[]
    for k in out:
        if 'decode' in k:
            result.append(out[k].view(-1,85))
    result=torch.cat(result,0).detach().numpy()
    bboxes=convert_pred(result,(h,w))
    for bb in bboxes:
        x,y,x1,y1=bb.astype(np.int)[:4]
        cv2.rectangle(img,(x,y),(x1,y1),(255,0,0),3)
    cv2.imshow('',img)
    if cv2.waitKey(0)&0xff ==27:
        break
cv2.destroyAllWindows()
