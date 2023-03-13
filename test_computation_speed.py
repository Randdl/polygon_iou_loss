from fixed_polygon_iou_loss import batch_poly_diou_loss, batch_poly_iou, batch_unconvex_poly_iou
import time
import numpy as np
import torch
import cv2


def raster_poly_IOU(poly1, poly2, scale=1000):
    poly1 = (poly1 * scale / 2.0 + scale / 4.0).astype(np.int32)
    poly2 = (poly2 * scale / 2.0 + scale / 4.0).astype(np.int32)

    im1 = np.zeros([scale, scale])
    im2 = np.zeros([scale, scale])
    imi = np.zeros([scale, scale])

    im1 = cv2.fillPoly(im1, [poly1], color=1)
    im2 = cv2.fillPoly(im2, [poly2], color=1)
    imi = (im1 + im2) / 2.0

    imi = np.floor(imi)

    ai = np.sum(imi)
    a1 = np.sum(im1)
    a2 = np.sum(im2)

    iou = (ai) / (a1 + a2 - ai)

    return iou


size = 128
poly1 = np.array([[124.35, 46.53], [253.35, 21.31], [333.31, 214.3], [134.31, 11.3]])
poly1 = poly1 / 1000
poly2 = np.array([[108.35, 53.53], [203.35, 234.31], [34.31, 24.3], [666.31, 126.3]])
poly2 = poly2 / 1000
start = time.time()
raster_poly_IOU(poly1, poly2)
raster_time = time.time() - start
print(raster_time)
polygon = torch.tensor([[124.35, 46.53], [253.35, 21.31], [333.31, 214.3], [134.31, 11.3]], dtype=torch.float)
polygon = polygon.repeat(size, 1, 1)
polygon3 = torch.tensor([[108.35, 53.53], [203.35, 234.31], [34.31, 24.3], [666.31, 126.3]], dtype=torch.float)
polygon3 = polygon3.repeat(size, 1, 1)
start = time.time()
iou = batch_poly_iou(polygon, polygon3)
print(iou)

batched_time = time.time() - start
print(batched_time)
print(raster_time / batched_time * size)
# iou = batch_unconvex_poly_iou(polygon, polygon3)
# print(iou)
