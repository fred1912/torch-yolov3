import numpy as np
import imgaug as ia
from imgaug import augmenters as iaa

class Compose(object):
    """Composes several transforms together.
    Args:
        transforms (list of ``Transform`` objects): list of transforms to compose.
    """
    def __init__(self, transforms=[]):
        self.transforms = transforms

    def __call__(self, img):
        for t in self.transforms:
            img = t(img)
        return img

    def add(self, transform):
        self.transforms.append(transform)

class KeepAspect(object):
    def __init__(self):
        pass

    def __call__(self, sample):
        image, label = sample['image'], sample['bboxes']
        h, w, _ = image.shape
        dim_diff = np.abs(h - w)
        pad1, pad2 = dim_diff // 2, dim_diff - dim_diff // 2
        pad = ((pad1, pad2), (0, 0), (0, 0)) if h <= w else ((0, 0), (pad1, pad2), (0, 0))
        image_new = np.pad(image, pad, 'constant', constant_values=0)
        padded_h, padded_w, _ = image_new.shape
        for i in range(len(label)):
            label[i].x1 += pad[1][0]
            label[i].y1 += pad[0][0]
            label[i].x2 += pad[1][0]
            label[i].y2 += pad[0][0]
        return {'image': image_new, 'bboxes': label}


class ImageAug(object):
    def __init__(self,height,width,split):
        sometimes = lambda aug: iaa.Sometimes(0.5, aug)
        if split == 'train':
            self.seq = iaa.Sequential(
                [
                    # Blur each image with varying strength using
                    # gaussian blur (sigma between 0 and 3.0),
                    # average/uniform blur (kernel size between 2x2 and 7x7)
                    # median blur (kernel size between 3x3 and 11x11).
                    iaa.Fliplr(0.5), # horizontally flip 50% of all images
                    # iaa.Flipud(0.2), # vertically flip 20% of all images
                    # crop images by -5% to 10% of their height/width
                    sometimes(iaa.CropAndPad(
                        percent=(-0.05, 0.1),
                        pad_mode=ia.ALL,
                        pad_cval=(0, 255)
                    )),
                    sometimes(iaa.Affine(
                        scale={"x": (0.8, 1.2), "y": (0.8, 1.2)}, # scale images to 80-120% of their size, individually per axis
                        translate_percent={"x": (-0.2, 0.2), "y": (-0.2, 0.2)}, # translate by -20 to +20 percent (per axis)
                        # rotate=(-45, 45), # rotate by -45 to +45 degrees
                        # shear=(-16, 16), # shear by -16 to +16 degrees
                        order=[0, 1], # use nearest neighbour or bilinear interpolation (fast)
                        cval=(0, 255), # if mode is constant, use a cval between 0 and 255
                        mode=ia.ALL # use any of scikit-image's warping modes (see 2nd image from the top for examples)
                    )),
                    iaa.Resize({"height": height, "width": width}),
                    sometimes(iaa.OneOf([
                        iaa.GaussianBlur((0, 3.0)),
                        iaa.AverageBlur(k=(2, 7)),
                        iaa.MedianBlur(k=(3, 11)),
                    ])),
                    # Sharpen each image, overlay the result with the original
                    # image using an alpha between 0 (no sharpening) and 1
                    # (full sharpening effect).
                    sometimes(iaa.Sharpen(alpha=(0, 0.5), lightness=(0.75, 1.5))),
                    # Add gaussian noise to some images.
                    sometimes(iaa.AdditiveGaussianNoise(loc=0, scale=(0.0, 0.05*255), per_channel=0.5)),
                    # Add a value of -5 to 5 to each pixel.
                    sometimes(iaa.Add((-5, 5), per_channel=0.5)),
                    # Change brightness of images (80-120% of original value).
                    sometimes(iaa.Multiply((0.8, 1.2), per_channel=0.5)),
                    # Improve or worsen the contrast of images.
                    sometimes(iaa.ContrastNormalization((0.5, 2.0), per_channel=0.5)),
                ],
                # do all of the above augmentations in random order
                random_order=True
            )
        else:
            self.seq=iaa.Resize({"height": height, "width": width})

    def __call__(self, sample):
        seq_det = self.seq.to_deterministic()
        image, label = sample['image'], sample['bboxes']
        shape=image.shape
        image = seq_det.augment_image(image)
        bbs = ia.BoundingBoxesOnImage(label, shape=shape)
        label=seq_det.augment_bounding_boxes(bbs).bounding_boxes
        return {'image': image, 'bboxes': label}