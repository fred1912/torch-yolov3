import numpy as np
import pycocotools.coco as coco

class YOLO_Kmeans:

    def __init__(self, cluster_number,filename):
        self.cluster_number = cluster_number
        self.filename = filename

    def iou(self, boxes, clusters):  # 1 box -> k clusters
        n = boxes.shape[0]
        k = self.cluster_number

        box_area = boxes[:, 0] * boxes[:, 1]
        box_area = box_area.repeat(k)
        box_area = np.reshape(box_area, (n, k))

        cluster_area = clusters[:, 0] * clusters[:, 1]
        cluster_area = np.tile(cluster_area, [1, n])
        cluster_area = np.reshape(cluster_area, (n, k))

        box_w_matrix = np.reshape(boxes[:, 0].repeat(k), (n, k))
        cluster_w_matrix = np.reshape(np.tile(clusters[:, 0], (1, n)), (n, k))
        min_w_matrix = np.minimum(cluster_w_matrix, box_w_matrix)

        box_h_matrix = np.reshape(boxes[:, 1].repeat(k), (n, k))
        cluster_h_matrix = np.reshape(np.tile(clusters[:, 1], (1, n)), (n, k))
        min_h_matrix = np.minimum(cluster_h_matrix, box_h_matrix)
        inter_area = np.multiply(min_w_matrix, min_h_matrix)

        result = inter_area / (box_area + cluster_area - inter_area)
        return result

    def avg_iou(self, boxes, clusters):
        accuracy = np.mean([np.max(self.iou(boxes, clusters), axis=1)])
        return accuracy

    def kmeans(self, boxes, k, dist=np.median):
        box_number = boxes.shape[0]
        distances = np.empty((box_number, k))
        last_nearest = np.zeros((box_number,))
        np.random.seed()
        clusters = boxes[np.random.choice(
            box_number, k, replace=False)]  # init k clusters
        while True:
            distances = 1 - self.iou(boxes, clusters)
            current_nearest = np.argmin(distances, axis=1)
            if (last_nearest == current_nearest).all():
                break  # clusters won't change
            for cluster in range(k):
                clusters[cluster] = dist(  # update clusters
                    boxes[current_nearest == cluster], axis=0)
            last_nearest = current_nearest
        return clusters

    def json2boxes(self):
        print('==> load {} data.'.format(self.filename ))
        self.coco = coco.COCO(self.filename )
        self.images = self.coco.getImgIds()
        self.num_samples = len(self.images)
        print('Loaded {} samples'.format(self.num_samples))
        dataSet = []
        for img_id in self.images:
            image_detail = self.coco.loadImgs(ids=[img_id])[0]
            height,width = image_detail['height'],image_detail['width']
            ann_ids = self.coco.getAnnIds(imgIds=[img_id])
            anns = self.coco.loadAnns(ids=ann_ids)
            for k in range(len(anns)):
                ann = anns[k]
                w,h = ann['bbox'][2:]
                dataSet.append([w/width, h/height])
        result = np.array(dataSet)
        return result

    def json2clusters(self):
        all_boxes = self.json2boxes()
        result = self.kmeans(all_boxes, k=self.cluster_number)
        result = result[np.lexsort(result.T[0, None])]
        print("Accuracy: {:.2f}%".format(
            self.avg_iou(all_boxes, result) * 100))
        result = np.reshape(np.around(result,5),[self.cluster_number//3,3,2]).tolist()
        print("K anchors:\n {}".format(result))


if __name__ == "__main__":
    cluster_number = 9
    filename = '/data/DataSet/person_helmet/coco2017/annotations/instances_train2017.json'
    kmeans = YOLO_Kmeans(cluster_number, filename)
    kmeans.json2clusters()