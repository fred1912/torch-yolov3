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
    boxes1 = torch.cat([boxes1[..., :2] - boxes1[..., 2:] * 0.5,
                        boxes1[..., :2] + boxes1[..., 2:] * 0.5], dim=-1)
    boxes2 = torch.cat([boxes2[..., :2] - boxes2[..., 2:] * 0.5,
                        boxes2[..., :2] + boxes2[..., 2:] * 0.5], dim=-1)
    area1 = (boxes1[..., 2] - boxes1[..., 0]) * (boxes1[..., 3] - boxes1[..., 1])
    area2 = (boxes2[..., 2] - boxes2[..., 0]) * (boxes2[..., 3] - boxes2[..., 1])
    inter_max_xy = torch.min(boxes1[..., 2:], boxes2[..., 2:])
    inter_min_xy = torch.max(boxes1[..., :2], boxes2[..., :2])
    inter = torch.clamp((inter_max_xy - inter_min_xy), min=0)
    inter_area = inter[..., 0] * inter[..., 1]
    union = area1 + area2 - inter_area
    ious = inter_area / union
    ious = torch.clamp(ious, min=0., max=1.0)
    return ious

def cal_GIOU(boxes1, boxes2):
    boxes1 = torch.cat([boxes1[..., :2] - boxes1[..., 2:] * 0.5,
                        boxes1[..., :2] + boxes1[..., 2:] * 0.5], dim=-1)
    boxes2 = torch.cat([boxes2[..., :2] - boxes2[..., 2:] * 0.5,
                        boxes2[..., :2] + boxes2[..., 2:] * 0.5], dim=-1)
    area1 = (boxes1[..., 2] - boxes1[..., 0]) * (boxes1[..., 3] - boxes1[..., 1])
    area2 = (boxes2[..., 2] - boxes2[..., 0]) * (boxes2[..., 3] - boxes2[..., 1])
    inter_max_xy = torch.min(boxes1[..., 2:],boxes2[..., 2:])
    inter_min_xy = torch.max(boxes1[..., :2],boxes2[..., :2])
    out_max_xy = torch.max(boxes1[..., 2:],boxes2[..., 2:])
    out_min_xy = torch.min(boxes1[..., :2],boxes2[..., :2])
    inter = torch.clamp((inter_max_xy - inter_min_xy), min=0)
    inter_area = inter[..., 0] * inter[..., 1]
    outer = torch.clamp((out_max_xy - out_min_xy), min=0)
    outer_area = outer[..., 0] * outer[..., 1]
    union = area1+area2-inter_area
    closure = outer_area
    ious = inter_area / union - (closure - union) / closure
    ious = torch.clamp(ious,min=-1.0,max = 1.0)
    return ious

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
            if torch.isinf(GIOU_loss_):
                raise Exception("inf in GIOU_loss_")
            if torch.isnan(GIOU_loss_):
                raise Exception("nan in GIOU_loss_")
            if torch.isinf(prob_loss_):
                raise Exception("inf in prob_loss_")
            if torch.isnan(prob_loss_):
                raise Exception("nan in prob_loss_")
            if torch.isinf(conf_loss_):
                raise Exception("inf in conf_loss_")
            if torch.isnan(conf_loss_):
                raise Exception("nan in conf_loss_")
            GIOU_loss=GIOU_loss+GIOU_loss_
            prob_loss=prob_loss+prob_loss_
            conf_loss=conf_loss+conf_loss_
        loss=GIOU_loss + prob_loss + conf_loss
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
        bbox_loss_scale = 2.0 - 1.0 * label_xywh[:, :, :, :, 2:3] * label_xywh[:, :, :, :, 3:4] / (416. ** 2)

        GIOU_loss = torch.sum(bbox_loss_scale * obj_conf * (1.0 - GIOU))/B

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

class yoloLoss_2(nn.Module):
    def __init__(self):
        super(yoloLoss_2,self).__init__()
        self.iou_loss_thresh=0.5
        self.alpha =1
        self.gamma =2

    def forward(self, output, batch):
        loss,IOU_loss, prob_loss, conf_loss = 0., 0., 0., 0.
        for i in range(3):
            IOU_loss_, prob_loss_, conf_loss_ =self.loss_scale(output['yolo-%d'%i],output['yolo-%d-decode'%i],batch['yolo-%d-label'%i],batch['yolo-%d-xywh'%i])
            IOU_loss=IOU_loss+IOU_loss_
            prob_loss=prob_loss+prob_loss_
            conf_loss=conf_loss+conf_loss_
        loss=5.*IOU_loss + prob_loss + conf_loss
        loss_state={'loss':loss,'xywh_loss':IOU_loss,'conf_loss':conf_loss,'prob_loss':prob_loss}
        return loss,loss_state

    def loss_scale(self,conv,pred,label,bbox_xywh):
        conv = conv.permute(0, 2, 3, 1).contiguous() #  B,H,W,C
        B,H,W,C=conv.size()
        conv = conv.view(B,H,W,3,C//3)
        conv_dxdy = conv[:, :, :, :, 0:2]
        conv_dwdh = conv[:, :, :, :, 2:4]
        conv_conf = conv[:, :, :, :, 4:5]
        conv_prob = conv[:, :, :, :, 5:]

        pred_xywh = pred[:, :, :, :, 0:4]

        label_dxdy = label[:, :, :, :, 0:2]
        label_dwdh = label[:, :, :, :, 2:4]
        label_wh = label[:, :, :, :, 4:6]
        label_conf = label[:, :, :, :, 6:7]
        label_prob = label[:, :, :, :, 7:]
        bbox_loss_scale = 2.0 - 1.0 * label_wh[:, :, :, :, 0:1] * label_wh[:, :, :, :, 1:2] /(H*W)
        xy_loss = (bbox_loss_scale*label_conf *  torch.mean(F.binary_cross_entropy(_sigmoid(conv_dxdy), label_dxdy,reduce=False),dim= -1,keepdim=True)).sum()/B
        wh_loss = (bbox_loss_scale*label_conf * torch.pow(conv_dwdh - label_dwdh,2)/2.).sum() /B
        iou = cal_IOU(pred_xywh[:, :, :, :, np.newaxis, :], bbox_xywh[:, np.newaxis, np.newaxis, np.newaxis, :, :])
        iou = torch.max(iou, dim=-1, keepdim=True)[0]
        obj_bg = (1. - label_conf) * iou.lt(self.iou_loss_thresh).float()  # iou>=thresh 的 ct 将会被忽略
        conf_loss = self.__focal_loss(_sigmoid(conv_conf), label_conf, obj_bg).sum() / B
        prob_loss = (label_conf *  torch.mean(F.binary_cross_entropy(_sigmoid(conv_prob), label_prob,reduce=False),dim= -1,keepdim=True)).sum()/B
        IOU_loss=xy_loss+wh_loss
        return IOU_loss,prob_loss,conf_loss

    def __focal_loss(self,pred_conf, obj_conf , obj_bg):

        pos_inds = obj_conf.eq(1).float()
        neg_inds = obj_bg.eq(1).float()

        loss = 0
        pos_loss = torch.log(pred_conf) * torch.pow(1 - pred_conf, self.gamma) * pos_inds
        neg_loss = torch.log(1 - pred_conf) * torch.pow(pred_conf, self.gamma) * neg_inds
        num_pos = pos_inds.float().sum()

        if num_pos == 0:
            loss = loss - neg_loss
        else:
            loss = loss - (pos_loss + neg_loss)
        return loss


if __name__ == '__main__':
    from dataset.yolo_coco import yoloCOCO
    from torch.utils.data import DataLoader
    from models.yolo_head import Yolodet
    import darkyolo_config
    loader=DataLoader(yoloCOCO(darkyolo_config,'val'),batch_size=4,shuffle=True)
    model = Yolodet(darkyolo_config)
    crit=yoloLoss()
    for batch in loader:
        out = model(batch['input'])
        loss, loss_state=crit(out,batch)
        loss.backward()
        for k in loss_state:
            print('{} : {}  '.format(k,loss_state[k].item()),end='')

