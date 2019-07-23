from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from utils.torch_utils import _sigmoid,cal_IOU,cal_GIOU

class yoloLoss(nn.Module):
    def __init__(self):
        super(yoloLoss,self).__init__()
        self.iou_loss_thresh=0.5
        self.alpha =1
        self.gamma =2
        self.mse_loss = nn.MSELoss(reduce=False)
        self.bce_loss = nn.BCELoss(reduction='sum')

    def forward(self, output, batch):
        loss,IOU_loss, prob_loss, conf_loss = 0., 0., 0., 0.
        for i in range(len(output) // 2):
            IOU_loss_, prob_loss_, conf_loss_ =self.loss_scale(output['yolo-%d'%i],output['yolo-%d-decode'%i],batch['yolo-%d-label'%i],batch['yolo-%d-xywh'%i])
            IOU_loss=IOU_loss+IOU_loss_
            prob_loss=prob_loss+prob_loss_
            conf_loss=conf_loss+conf_loss_
        loss=5.0 * IOU_loss + prob_loss + conf_loss
        loss_state={'loss':loss,'xywh_loss':IOU_loss,'conf_loss':conf_loss,'prob_loss':prob_loss}
        return loss,loss_state

    def loss_scale(self,conv,pred,label,bbox_xywh):
        conv = conv.permute(0, 2, 3, 1).contiguous() #  B,H,W,C
        B,H,W,C=conv.size()
        conv = conv.view(B,H,W,3,C//3)

        label_conf = label[:, :, :, :, 6]
        conv_conf = _sigmoid(conv[:, :, :, :, 4])

        pred_xywh = pred[:, :, :, :, 0:4]

        mask = label_conf == 1

        label = label[mask]
        conv  = conv[mask]

        bbox_loss_scale = 2.0 - 1.0 * label[..., 4] * label[..., 5]
        loss_x = self.bce_loss(_sigmoid(conv[...,0]), label[...,0])
        loss_y = self.bce_loss(_sigmoid(conv[...,1]), label[...,1])
        loss_w = torch.sum(self.mse_loss(conv[...,2], label[...,2])*bbox_loss_scale)
        loss_h = torch.sum(self.mse_loss(conv[...,3], label[...,3])*bbox_loss_scale)

        iou = cal_IOU(pred_xywh[:, :, :, :, np.newaxis, :].detach(), bbox_xywh[:, np.newaxis, np.newaxis, np.newaxis, :, :])
        iou = torch.max(iou, dim=-1)[0]
        no_obj_mask = ((1. - label_conf) * iou.lt(self.iou_loss_thresh).float())==1  # iou>=thresh 的 ct 将会被忽略
        conf_loss = self.__focal_loss(conv_conf, label_conf, mask,no_obj_mask)/B
        prob_loss = self.bce_loss(_sigmoid(conv[...,5:]), label[...,7:])/B
        IOU_loss=(loss_x+loss_y+loss_w+loss_h)/B

        return IOU_loss,prob_loss,conf_loss

    def __focal_loss(self,pred_conf, obj_conf , obj_mask,no_obj_mask):
        #focal= self.alpha*torch.pow(torch.abs(pred_conf-obj_conf),self.gamma)
        focal= 1.0
        loss = focal*(self.bce_loss(pred_conf[obj_mask], obj_conf[obj_mask])
                      + 0.1* self.bce_loss(pred_conf[no_obj_mask], obj_conf[no_obj_mask]))
        return loss


class yoloLoss_GIOU(nn.Module):
    def __init__(self):
        super(yoloLoss_GIOU,self).__init__()
        self.iou_loss_thresh=0.5
        self.alpha =1
        self.gamma =2
        self.bce_loss = nn.BCELoss(reduction='sum')
    def forward(self, output, batch):
        loss,GIOU_loss, prob_loss, conf_loss = 0., 0., 0., 0.
        for i in range(len(output)//2):
            GIOU_loss_, prob_loss_, conf_loss_ =self.loss_scale(output['yolo-%d-decode'%i],batch['yolo-%d-label'%i],batch['yolo-%d-xywh'%i])
            if torch.isinf(GIOU_loss_):
                raise Exception("inf in GIOU_loss_")
            if torch.isnan(GIOU_loss_):
                raise Exception("nan in GIOU_loss_")
            GIOU_loss=GIOU_loss+GIOU_loss_
            prob_loss=prob_loss+prob_loss_
            conf_loss=conf_loss+conf_loss_
        loss=5* GIOU_loss + prob_loss + conf_loss
        loss_state={'loss':loss,'xywh_loss':GIOU_loss,'conf_loss':conf_loss,'prob_loss':prob_loss}
        return loss,loss_state

    def loss_scale(self,pred,label,bbox_xywh):
        B,H,W,an,C=pred.size()
        obj_conf = label[:, :, :, :, 6:7]
        pred_conf = pred[:, :, :, :, 4:5]
        pred_xywh = pred[:, :, :, :, 0:4]
        mask = obj_conf[...,0]==1
        label = label[mask]
        pred = pred[mask]

        GIOU=cal_GIOU(pred[...,0:4],label[...,0:4])
        bbox_loss_scale = 2.0 - 1.0 * label[...,4] * label[...,5]
        GIOU_loss = torch.sum(bbox_loss_scale*(1.0 - GIOU))/B

        iou = cal_IOU(pred_xywh[:,:,:,:,np.newaxis,:].detach(),bbox_xywh[:,np.newaxis,np.newaxis,np.newaxis,:,:])
        iou=torch.max(iou,dim=-1,keepdim=True)[0]
        obj_bg=(1.-obj_conf)* iou.lt(self.iou_loss_thresh).float()  # iou>=thresh 的 ct 将会被忽略

        conf_loss = self.__focal_loss(pred_conf,obj_conf,obj_bg).sum()/B

        prob_loss = self.bce_loss(pred[...,5:],label[...,7:])/B
        return GIOU_loss,prob_loss,conf_loss

    def __focal_loss(self,pred_conf, obj_conf , obj_bg):
        focal= self.alpha*torch.pow(torch.abs(pred_conf-obj_conf),self.gamma)
        loss = focal*(obj_conf*F.binary_cross_entropy(pred_conf, obj_conf,reduce=False)
                      + obj_bg*F.binary_cross_entropy(pred_conf, obj_conf,reduce=False))
        return loss

if __name__ == '__main__':
    from dataset import yoloCOCO,yoloPascal
    from torch.utils.data import DataLoader
    from models.yolo_head import Yolodet
    from config import hrnet_yolo,dark53_yolo
    from utils.checkpoint import load_checkpoint
    loader=DataLoader(yoloCOCO(dark53_yolo,'val'),batch_size=4,shuffle=False)
    model = Yolodet(dark53_yolo,pretrained=True)
    load_checkpoint(model,'../weights/pretrained/this_imple_yolov3.pth')
    crit=yoloLoss()
    for batch in loader:
        out = model(batch['input'])
        loss, loss_state=crit(out,batch)
        loss.backward()
        for k in loss_state:
            print('{} : {}  '.format(k,loss_state[k].item()),end='')
        print()