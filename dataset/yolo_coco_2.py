from dataset.coco import COCO
import cv2
import os
from utils.image import flip, color_aug
from utils.image import get_affine_transform, affine_transform
from utils.avg import cal_iou
import numpy as np

class yoloCOCO(COCO):

    def __init__(self,config, split):
        super(yoloCOCO,self).__init__(config, split)

    def __getitem__(self, index):
        img_id = self.images[index]
        file_name = self.coco.loadImgs(ids=[img_id])[0]['file_name']
        img_path = os.path.join(self.img_dir, file_name)
        ann_ids = self.coco.getAnnIds(imgIds=[img_id])
        anns = self.coco.loadAnns(ids=ann_ids)
        num_objs = min(len(anns), self.max_objs)

        img = cv2.imread(img_path)

        height, width = img.shape[0], img.shape[1]
        c = np.array([img.shape[1] / 2., img.shape[0] / 2.], dtype=np.float32)  # the center of imgae
        if self.config.keep_res:  # keep the shape of image
            input_h = (height | self.config.pad) + 1
            input_w = (width | self.config.pad) + 1
            s = np.array([input_w, input_h], dtype=np.float32)
        else:
            s = max(img.shape[0], img.shape[1]) * 1.0  # the max scale of image
            input_h, input_w = self.config.input_h, self.config.input_w

        flipped = False
        if self.split == 'train':
            if not self.config.not_rand_crop:  # random crop
                s = s * np.random.choice(np.arange(0.6, 1.4, 0.1))
                w_border = self._get_border(128, img.shape[1])
                h_border = self._get_border(128, img.shape[0])
                c[0] = np.random.randint(low=w_border, high=img.shape[1] - w_border)
                c[1] = np.random.randint(low=h_border, high=img.shape[0] - h_border)
            else:
                sf = self.config.scale
                cf = self.config.shift
                c[0] += s * np.clip(np.random.randn() * cf, -2 * cf, 2 * cf)
                c[1] += s * np.clip(np.random.randn() * cf, -2 * cf, 2 * cf)
                s = s * np.clip(np.random.randn() * sf + 1, 1 - sf, 1 + sf)

            if np.random.random() < self.config.flip:
                flipped = True
                img = img[:, ::-1, :]
                c[0] = width - c[0] - 1

        trans_input = get_affine_transform(
            c, s, 0, [input_w, input_h])
        inp = cv2.warpAffine(img, trans_input,
                             (input_w, input_h),
                             flags=cv2.INTER_LINEAR)
        image = np.copy(inp)

        inp = (inp.astype(np.float32) / 255.)
        if self.split == 'train' and not self.config.no_color_aug:
            color_aug(self._data_rng, inp, self._eig_val, self._eig_vec)
        inp = (inp - self.mean) / self.std
        inp = inp.transpose(2, 0, 1)

        output_hs = input_h // np.array(self.config.DOWN_RATION)
        output_ws = input_w //  np.array(self.config.DOWN_RATION)

        num_classes = self.num_classes
        anchors_num=len(self.config.ANCHOR)
        anchors=np.array(self.config.ANCHOR)

        label = [np.zeros((output_h, output_w,anchors_num, 7 + num_classes), dtype=np.float32) for  output_w, output_h in zip(output_ws,output_hs)]
        bboxs_xywh = [np.zeros((self.max_objs, 4), dtype=np.float32) for _ in range(len(self.config.DOWN_RATION))]
        bbox_count = np.zeros((3,))
        for k in range(num_objs):
            ann = anns[k]
            bbox = self._coco_box_to_bbox(ann['bbox'])
            cls_id = int(self.cat_ids[ann['category_id']])
            if flipped:
                bbox[[0, 2]] = width - bbox[[2, 0]] - 1

            bbox[ :2] = affine_transform(bbox[:2], trans_input)
            bbox[ 2:] = affine_transform(bbox[2:], trans_input)
            bbox[[0, 2]] = np.clip(bbox[[0, 2]], 0, input_w- 1)
            bbox[[1, 3]] = np.clip(bbox[[1, 3]], 0, input_h - 1)
            bbox_xywh = np.concatenate(
                [(bbox[2:] + bbox[ :2]) * 0.5, bbox[ 2:] - bbox[ :2]],axis=-1)
            onehot = np.zeros(num_classes, dtype=np.float)
            onehot[cls_id] = 1.0
            uniform_distribution = np.full(num_classes, 1.0 / num_classes)
            deta = 0.01
            smooth_onehot = onehot * (1 - deta) + deta * uniform_distribution
            iou = []
            exist_positive = False
            bbox_xywh_scale = bbox_xywh[np.newaxis, :] / np.array(self.config.DOWN_RATION)[:, np.newaxis]
            for i in range(len(self.config.DOWN_RATION)):
                anchors_xywh = np.zeros((anchors_num, 4))
                anchors_xywh[:, 0:2] = np.floor(bbox_xywh_scale[i,0:2]).astype(np.int32) + 0.5
                anchors_xywh[:, 2:4] = anchors[i]
                iou_temp=cal_iou(bbox_xywh_scale[i,np.newaxis,:],anchors_xywh)
                iou.append(iou_temp)
                iou_mask = iou_temp > 0.3
                if np.any(iou_mask):
                    xind, yind = np.floor(bbox_xywh_scale[i,0:2]).astype(np.int32)
                    dxdy=bbox_xywh_scale[i, 0:2]-[xind,yind]
                    twth=np.log(bbox_xywh_scale[i, 2:4]/anchors[i,iou_mask])
                    twth=np.where(np.isinf(twth),np.zeros_like(twth),twth)
                    dxdy=np.tile(dxdy,[twth.shape[0],1])
                    label[i][yind, xind, iou_mask, :] = 0.
                    label[i][yind, xind, iou_mask, 0:2] = dxdy
                    label[i][yind, xind, iou_mask, 2:4] = twth
                    label[i][yind, xind, iou_mask, 4:6] = bbox_xywh_scale[i, 2:4]
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
                xind, yind = np.floor( bbox_xywh_scale[best_detect,0:2]).astype(np.int32)
                dxdy = bbox_xywh_scale[best_detect, 0:2] - [xind, yind]
                twth = np.log(bbox_xywh_scale[best_detect, 2:4] / anchors[best_detect, best_anchor])
                twth = np.where(np.isinf(twth), np.zeros_like(twth), twth)
                label[best_detect][yind, xind, best_anchor, :] = 0.
                label[best_detect][yind, xind, best_anchor, 0:2] = dxdy
                label[best_detect][yind, xind, best_anchor, 2:4] = twth
                label[best_detect][yind, xind, best_anchor, 4:6] = bbox_xywh_scale[best_detect, 2:4]
                label[best_detect][yind, xind, best_anchor, 6:7] = 1.0
                label[best_detect][yind, xind, best_anchor, 7:] = smooth_onehot
                bbox_ind = int(bbox_count[best_detect] % self.max_objs)
                bboxs_xywh[best_detect][bbox_ind, :4] = bbox_xywh
                bbox_count[best_detect] += 1

        ret = {'input': inp}
        for id in range(3):
            ret.update({'yolo-%d-label'%(id):label[id]})
            ret.update({'yolo-%d-xywh' % (id): bboxs_xywh[id]})
        if  not self.split == 'train':
            ret['image'] = image
        return ret

if __name__ == '__main__':
    import yolo_config
    data = yoloCOCO(yolo_config, 'val')
    for i in range(1000):
        t = data[i]
        print(i)

