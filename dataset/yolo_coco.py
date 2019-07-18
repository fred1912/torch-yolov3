from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from dataset.base import COCO
import cv2
import os
from utils.transforms import *
from utils.util import cal_iou
import numpy as np
import imgaug as ia

class yoloCOCO(COCO):
    def __init__(self,config, split):
        super(yoloCOCO,self).__init__(config, split)
        self.use_giou=config.DATASET.USE_GIOU
        self.input_h = config.DATASET.INPUT_H
        self.input_w = config.DATASET.INPUT_W
        self.KeepAspect=KeepAspect()
        self.imgaug=ImageAug(self.input_h ,self.input_w ,split)

    def __getitem__(self, index):
        ret ,img_id= self.get_image_bboxes(index)
        ret = self.KeepAspect(ret)
        ret=self.imgaug(ret)
        inp,bboxes= ret['image'],ret['bboxes']
        inp = (inp.astype(np.float32) / 255.)
        inp = (inp - self.mean) / self.std
        inp = inp.transpose(2, 0, 1)
        output_hs = self.input_h // np.array(self.config.DOWN_RATION)
        output_ws = self.input_w // np.array(self.config.DOWN_RATION)
        num_classes = self.num_classes
        anchors_num = len(self.config.ANCHORS[0])
        num_objs = len(bboxes)
        anchors = np.array(self.config.ANCHORS) * np.reshape(np.array([self.input_w, self.input_h]), [1, 1, 2])

        label = [np.zeros((output_h, output_w, anchors_num, 5 + 2 + num_classes), dtype=np.float32) for
                 output_w, output_h in zip(output_ws, output_hs)]
        bboxs_xywh = [np.zeros((self.max_objs, 4), dtype=np.float32) for _ in range(len(self.config.DOWN_RATION))]
        bbox_count = np.zeros((len(self.config.DOWN_RATION),))
        for k in range(num_objs):
            ann = bboxes[k]
            onehot = np.zeros(num_classes, dtype=np.float)
            onehot[ann.label] = 1.0
            uniform_distribution = np.full(num_classes, 1.0 / num_classes)
            deta = 0.01
            smooth_onehot = onehot * (1 - deta) + deta * uniform_distribution
            iou = []
            exist_positive = False
            bbox = np.array([ann.x1, ann.y1, ann.x2, ann.y2])
            bbox[[0, 2]] = np.clip(bbox[[0, 2]], 0, self.input_w - 1)
            bbox[[1, 3]] = np.clip(bbox[[1, 3]], 0, self.input_h - 1)
            bbox_xywh = np.concatenate(
                [(bbox[2:] + bbox[:2]) * 0.5, bbox[2:] - bbox[:2]], axis=-1)
            if np.any(bbox_xywh == 0.):
                continue
            bbox_xywh_scale = bbox_xywh[np.newaxis, :] / np.array(self.config.DOWN_RATION)[:, np.newaxis]
            for i in range(len(self.config.DOWN_RATION)):
                anchors_xywh = np.zeros((anchors_num, 4))
                anchors_xywh[:, 0:2] = np.floor(bbox_xywh_scale[i, 0:2]).astype(np.int32) + 0.5
                anchors_xywh[:, 2:4] = anchors[i] / self.config.DOWN_RATION[i]
                iou_temp = cal_iou(bbox_xywh_scale[i, np.newaxis, :], anchors_xywh)
                iou.append(iou_temp)
                iou_mask = iou_temp > 0.3
                if np.any(iou_mask):
                    xind, yind = np.floor(bbox_xywh_scale[i, 0:2]).astype(np.int32)
                    label[i][yind, xind, iou_mask, :] = 0.

                    if self.use_giou:
                        label[i][yind, xind, iou_mask, 0:4] = bbox_xywh
                    else:
                        dxdy = bbox_xywh_scale[i, 0:2] - [xind, yind]
                        twth = np.log(bbox_xywh[2:4] / anchors[i, iou_mask])
                        twth = np.where(np.isinf(twth), np.zeros_like(twth), twth)
                        dxdy = np.tile(dxdy, [twth.shape[0], 1])
                        label[i][yind, xind, iou_mask, 0:2] = dxdy
                        label[i][yind, xind, iou_mask, 2:4] = twth

                    label[i][yind, xind, iou_mask, 4:6] = bbox_xywh[2:4] / np.array([self.input_w, self.input_h])
                    label[i][yind, xind, iou_mask, 6:7] = 1.0
                    label[i][yind, xind, iou_mask, 7:] = smooth_onehot
                    bbox_ind = int(bbox_count[i] % self.max_objs)
                    bboxs_xywh[i][bbox_ind, :4] = bbox_xywh
                    bbox_count[i] += 1
                    exist_positive = True

            if not exist_positive:
                best_anchor_ind = np.argmax(np.array(iou).reshape(-1), axis=-1)
                best_detect = int(best_anchor_ind / anchors_num)
                best_anchor = int(best_anchor_ind % anchors_num)
                xind, yind = np.floor(bbox_xywh_scale[best_detect, 0:2]).astype(np.int32)
                label[best_detect][yind, xind, best_anchor, :] = 0.

                if self.use_giou:
                    label[best_detect][yind, xind, best_anchor, 0:4] = bbox_xywh
                else:
                    dxdy = bbox_xywh_scale[best_detect, 0:2] - [xind, yind]
                    twth = np.log(bbox_xywh[2:4] / anchors[best_detect, best_anchor])
                    twth = np.where(np.isinf(twth), np.zeros_like(twth), twth)
                    label[best_detect][yind, xind, best_anchor, 0:2] = dxdy
                    label[best_detect][yind, xind, best_anchor, 2:4] = twth

                label[best_detect][yind, xind, best_anchor, 4:6] = bbox_xywh[2:4] / np.array([self.input_w, self.input_h])
                label[best_detect][yind, xind, best_anchor, 6:7] = 1.0
                label[best_detect][yind, xind, best_anchor, 7:] = smooth_onehot

                bbox_ind = int(bbox_count[best_detect] % self.max_objs)
                bboxs_xywh[best_detect][bbox_ind, :4] = bbox_xywh
                bbox_count[best_detect] += 1

        ret = {'input': inp}
        for id in range(len(self.config.DOWN_RATION)):
            ret.update({'yolo-%d-label' % (id): label[id]})
            ret.update({'yolo-%d-xywh' % (id): bboxs_xywh[id]})
        if self.split != 'train':
            ret.update({'img_id':img_id})
        return ret



    def get_image_bboxes(self,index):
        img_id = self.images[index]
        file_name = self.coco.loadImgs(ids=[img_id])[0]['file_name']
        img_path = os.path.join(self.img_dir, file_name)
        ann_ids = self.coco.getAnnIds(imgIds=[img_id])
        anns = self.coco.loadAnns(ids=ann_ids)
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        bboexs=[]
        for i,ann in enumerate(anns):
            bbox = self._coco_box_to_bbox(ann['bbox'])
            cls_id = int(self.cat_ids[ann['category_id']])
            bboexs.append(ia.BoundingBox(x1=bbox[0], y1=bbox[1], x2=bbox[2], y2=bbox[3],label=cls_id))
        return {'image':img,'bboxes':bboexs},img_id


if __name__ == '__main__':
    from config import hrnet_yolo
    data = yoloCOCO(hrnet_yolo, 'val')
    for i in range(1000):
        t = data[i]
        print(i)
        pass