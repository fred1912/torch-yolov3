from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

def _sigmoid(x):
  y = torch.clamp(x.sigmoid(), min=1e-4, max=1-1e-4)
  return y

def _gather_feat(feat, ind, mask=None):
    dim  = feat.size(2) #C=2
    ind  = ind.unsqueeze(2).expand(ind.size(0), ind.size(1), dim) # B,MAX_OBJ -> B,MAX_OBJ,1 -> B,MAX_OBJ,C=2
    feat = feat.gather(1, ind)  # B,MAX_OBJ,2
    if mask is not None:
        mask = mask.unsqueeze(2).expand_as(feat)
        feat = feat[mask]
        feat = feat.view(-1, dim)
    return feat

def _tranpose_and_gather_feat(feat, ind):
    feat = feat.permute(0, 2, 3, 1).contiguous() #  B,H,W,C
    feat = feat.view(feat.size(0), -1, feat.size(3)) # B,HXW,C
    feat = _gather_feat(feat, ind)  # B,MAX_OBJ,2
    return feat

class FocalLoss(nn.Module):

    def __init__(self,beta = 2 ,alpha = 0.8):

        super(FocalLoss,self).__init__()
        self.beta = beta
        self.alpha = alpha

    def forward(self, pred, target):

        pos_inds = target.eq(1).float()   # 高斯核中心 为 1
        neg_inds = target.lt(1).float()   # 高斯核其他地方 小于1

        neg_weights = torch.pow(1 - target, 4)  # 距离高斯核越远的中心的样本点， 越为hard example , 施加一定权重

        loss = 0

        pos_loss = torch.log(pred) * torch.pow(1 - pred, self.beta) * pos_inds  #高斯核中心的损失，(1-y')^beta*log(y')   yt=1
        neg_loss = torch.log(1 - pred) * torch.pow(pred, self.beta) * neg_weights * neg_inds  #  y'^beta*log(1-y')   yt=0

        num_pos = pos_inds.float().sum()   # 高斯核中心点个数，即样本个数
        pos_loss = pos_loss.sum()
        neg_loss = neg_loss.sum()

        if num_pos == 0:
            loss = loss - neg_loss
        else:
            loss = loss - (pos_loss + neg_loss) / num_pos
        return loss

class RegL1loss(nn.Module):
    def __init__(self):
        super(RegL1loss,self).__init__()

    def forward(self, output, mask, ind, target):
        pred = _tranpose_and_gather_feat(output, ind) # B,MAX_OBJ,2
        mask = mask.unsqueeze(2).expand_as(pred).float() # B,MAX_OBJ,2
        loss = F.l1_loss(pred * mask, target * mask, size_average=False) # ind=0 时 mask=0  即只有高斯核的中心允许往回传梯度
        loss = loss / (mask.sum() + 1e-4)  #mask.sum()  高斯核中心个数，即为正样本个数
        return loss

class Ctdetloss(nn.Module):

    def __init__(self,config):
        super(Ctdetloss,self).__init__()
        self.crit_hm = FocalLoss()
        self.crit_off = RegL1loss()
        self.crit_wh =  RegL1loss()
        self.hm_weight= config.TRAIN['HM_WEIGHT']
        self.wh_weight = config.TRAIN['WH_WEIGHT']
        self.off_weight = config.TRAIN['OFF_WEIGHT']

    def forward(self, output , batch):

        output['hm']=_sigmoid(output['hm'])
        hm_loss = self.crit_hm(output['hm'],batch['hm'])
        wh_loss = self.crit_wh(output['wh'], batch['reg_mask'],batch['ind'], batch['wh'])
        off_loss = self.crit_off(output['reg'], batch['reg_mask'],batch['ind'], batch['reg'])
        loss = self.hm_weight* hm_loss + self.wh_weight * wh_loss + self.off_weight * off_loss
        loss_stats= {'loss': loss, 'hm_loss': hm_loss,
                  'wh_loss': wh_loss, 'off_loss': off_loss}
        return loss,loss_stats


def cal_IOU(boxes1,boxes2):

    boxes1_area =  boxes1[..., 2] * boxes1[..., 3]
    boxes2_area =  boxes2[..., 2] * boxes2[..., 3]
    boxes1 = torch.cat([boxes1[..., :2] - boxes1[..., 2:] * 0.5,
                        boxes1[..., :2] + boxes1[..., 2:] * 0.5], dim=-1)
    boxes2 = torch.cat([boxes2[..., :2] - boxes2[..., 2:] * 0.5,
                        boxes2[..., :2] + boxes2[..., 2:] * 0.5], dim=-1)
    inter_left = torch.max(boxes1[...,:2],boxes2[...,:2])
    inter_right = torch.min(boxes1[...,2:], boxes2[...,2:])
    inter_wh = inter_right-inter_left
    inter_wh = inter_wh * inter_wh.ge(0).float()
    inter_area = inter_wh[...,0]*inter_wh[...,1]
    union_area = boxes1_area + boxes2_area - inter_area
    IOU = 1.0 * inter_area / union_area

    return IOU

def cal_GIOU(boxes1, boxes2):

    boxes1 = torch.cat([boxes1[..., :2] - boxes1[..., 2:] * 0.5,
                        boxes1[..., :2] + boxes1[..., 2:] * 0.5], dim=-1)
    boxes2 = torch.cat([boxes2[..., :2] - boxes2[..., 2:] * 0.5,
                        boxes2[..., :2] + boxes2[..., 2:] * 0.5], dim=-1)

    boxes1 = torch.cat([torch.min(boxes1[..., :2], boxes1[..., 2:]), torch.max(boxes1[..., :2], boxes1[..., 2:])],
                       dim=-1)
    boxes2 = torch.cat([torch.min(boxes2[..., :2], boxes2[..., 2:]), torch.max(boxes2[..., :2], boxes2[..., 2:])],
                       dim=-1)
    boxes1_area = (boxes1[..., 2] - boxes1[..., 0]) * (boxes1[..., 3] - boxes1[..., 1])
    boxes2_area = (boxes2[..., 2] - boxes2[..., 0]) * (boxes2[..., 3] - boxes2[..., 1])

    inter_left = torch.max(boxes1[...,:2],boxes2[...,:2])
    inter_right = torch.min(boxes1[...,2:], boxes2[...,2:])

    inter_wh = inter_right-inter_left
    inter_wh = inter_wh * inter_wh.ge(0).float()
    inter_area = inter_wh[...,0]*inter_wh[...,1]
    union_area = boxes1_area + boxes2_area - inter_area
    IOU = 1.0 * inter_area / union_area

    rh_left   = torch.min(boxes1[...,:2],boxes2[...,:2])
    rh_right  = torch.max(boxes1[...,2:], boxes2[...,2:])
    rh_inter_wh = rh_right - rh_left
    rh_inter_wh = rh_inter_wh * rh_inter_wh.ge(0).float()
    rn_area = rh_inter_wh[...,0]*rh_inter_wh[...,1]
    GIOU = IOU - 1.0 * (rn_area - union_area) / rn_area

    return GIOU


class yoloLoss(nn.Module):
    def __init__(self):
        super(yoloLoss,self).__init__()
        self.iou_loss_thresh=0.5
        self.alpha =1
        self.gamma =2
        self.BCEloss=nn.BCELoss(reduce=False)

    def forward(self, output, batch):
        loss,GIOU_loss, prob_loss, conf_loss = 0., 0., 0., 0.
        for i in range(3):
            GIOU_loss_, prob_loss_, conf_loss_ =self.loss_scale(output['yolo-%d-decode'%i],batch['yolo-%d-label'%i],batch['yolo-%d-xywh'%i])
            GIOU_loss=GIOU_loss+GIOU_loss_
            prob_loss=prob_loss+prob_loss_
            conf_loss=conf_loss+conf_loss_
        loss=5. * GIOU_loss + prob_loss + conf_loss
        loss_state={'loss':loss,'xywh_loss':GIOU_loss,'conf_loss':conf_loss,'prob_loss':prob_loss}
        return loss,loss_state

    def loss_scale(self,pred,label,bbox_xywh):

        B,H,W,an,C=pred.size()
        pred_xywh = pred[:, :, :, :, 0:4]
        pred_conf = pred[:, :, :, :, 4:5]
        pred_prob = pred[:, :, :, :, 5:]

        label_xywh = label[:, :, :, :, 0:4]
        obj_conf = label[:, :, :, :, 4:5]
        label_prob = label[:, :, :, :, 5:]

        GIOU=cal_GIOU(pred_xywh,label_xywh)
        GIOU = torch.unsqueeze(GIOU,-1)
        GIOU_loss = torch.sum(obj_conf * (1.0 - GIOU))/B

        iou = cal_IOU(pred_xywh[:,:,:,:,np.newaxis,:],bbox_xywh[:,np.newaxis,np.newaxis,np.newaxis,:,:])

        iou=torch.max(iou,dim=-1,keepdim=True)[0]

        obj_bg=(1.-obj_conf)* iou.lt(self.iou_loss_thresh).float()  # iou>=thresh 的 ct 将会被忽略

        conf_loss = self.__focal_loss(pred_conf,obj_conf,obj_bg).sum()/B

        prob_loss = torch.sum(obj_conf * torch.mean(self.BCEloss(pred_prob,label_prob),dim=-1,keepdim=True))/B


        return GIOU_loss,prob_loss,conf_loss

    def __focal_loss(self,pred_conf, obj_conf , obj_bg):

        focal= self.alpha*torch.pow(torch.abs(pred_conf-obj_conf),self.gamma)
        loss = focal*(obj_conf*F.binary_cross_entropy(pred_conf, obj_conf,reduce=False)
                      + 0.1 * obj_bg*F.binary_cross_entropy(pred_conf, obj_conf,reduce=False))
        return loss

if __name__ == '__main__':
    from dataset.yolo_coco import yoloCOCO
    from torch.utils.data import DataLoader
    from models.yolo_head import Yolodet
    import yolo_config
    loader=DataLoader(yoloCOCO(yolo_config,'val'),batch_size=4,shuffle=True)
    model = Yolodet(yolo_config)
    crit=yoloLoss()
    for batch in loader:
        out = model(batch['input'])
        loss, loss_state=crit(out,batch)
        loss.backward()
        for k in loss_state:
            print('{} : {}  '.format(k,loss_state[k].item()),end='')
        print()