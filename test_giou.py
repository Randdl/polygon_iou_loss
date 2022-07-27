import torch
import cv2
from polyogn_iou_loss import *
from matplotlib import pyplot as plt
import numpy as np

def plot_poly(poly, color=(0, 0, 255), im=None, lines=True, show=True, text=None):
    if im is None:
        s = 1000
        im = np.zeros([s, s, 3]) + 255
    else:
        s = im.shape[0]

    if len(poly) > 0:
        poly = poly * s / 2.0 + s / 4.0

    for p_idx, point in enumerate(poly):
        point = point.int()
        im = cv2.circle(im, (point[0], point[1]), 3, color, -1)

    if lines:
        for i in range(-1, len(poly) - 1):
            p1 = poly[i].int()
            p2 = poly[i + 1].int()
            im = cv2.line(im, (p1[0], p1[1]), (p2[0], p2[1]), color, 1)

    if text is not None:
        im = cv2.putText(im, text, (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 1)

    if show:
        cv2.imshow("im", im)
        cv2.waitKey(500)
        cv2.destroyAllWindows()
    return im


def plot_adjacency(points, adjacency, color=(0, 0, 255), im=None):
    if im is None:
        s = 1000
        im = np.zeros([s, s, 3]) + 255
    else:
        s = im.shape[0]

    if len(points) > 0:
        points = points * s / 2.0 + s / 4.0

    for i in range(adjacency.shape[0]):
        p1 = points[i, :].int()

        for j in range(adjacency.shape[1]):
            if adjacency[i, j] == 1:
                p2 = points[j, :].int()
                im = cv2.line(im, (p1[0], p1[1]), (p2[0], p2[1]), color, 2)
        im = cv2.putText(im, str(i), (p1[0], p1[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 0, 0), 1)

    cv2.imshow("im", im)
    cv2.waitKey(0)
    # cv2.destroyAllWindows()
    return

variance = 150
init_mid_np = np.concatenate((np.random.normal(150, variance / 3, size=(1, 2)), np.random.normal(0, variance, size=(2, 2))), axis=0)
init_mid = torch.Tensor(init_mid_np)
init_mid2 = torch.Tensor(init_mid_np)
target_mid_np = np.concatenate((np.random.normal(150, variance / 3, size=(1, 2)), np.random.normal(0, variance, size=(2, 2))), axis=0)
target_mid = torch.Tensor(target_mid_np)
# print(mid_2_points(init_mid))

# poly = torch.autograd.Variable(poly1, requires_grad=True)
poly_init = torch.autograd.Variable(init_mid, requires_grad=True)
poly_init2 = torch.autograd.Variable(init_mid2, requires_grad=True)
poly_target = mid_2_points(target_mid)

opt = torch.optim.Adam([poly_init], lr=0.55)
# scheduler = torch.optim.lr_scheduler.StepLR(opt, step_size=30, gamma=0.1)
opt2 = torch.optim.Adam([poly_init2], lr=0.55)
# scheduler2 = torch.optim.lr_scheduler.StepLR(opt2, step_size=30, gamma=0.1)

loss_history_1 = []
loss_history_2 = []

for k in range(1000):
    giou = c_poly_giou(mid_2_points(poly_init), poly_target)
    print(mid_2_points(poly_init))
    print(poly_target)
    opt.zero_grad()
    loss = c_poly_diou_loss(mid_2_points(poly_init), poly_target)
    loss_history_1.append(c_poly_iou(mid_2_points(poly_init), poly_target).detach().numpy())
    loss.backward()
    print("Polygon IOU: {}. lr:{}".format(giou.item(), opt.param_groups[0]['lr']))
    # im = np.zeros([1000, 1000, 3]) + 255
    # poly_1_detach = mid_2_points(poly_init).detach()
    # for i in range(-1, len(poly_1_detach) - 1):
    #     p1 = poly_1_detach[i].int()
    #     p2 = poly_1_detach[i + 1].int()
    #     im = cv2.line(im, (p1[0], p1[1]), (p2[0], p2[1]), (0, 0, 255), 1)
    # for i in range(-1, len(poly_target) - 1):
    #     p1 = poly_target[i].int()
    #     p2 = poly_target[i + 1].int()
    #     im = cv2.line(im, (p1[0], p1[1]), (p2[0], p2[1]), (255, 0, 255), 1)
    # im = plot_poly(mid_2_points(poly_init), color=(0, 0, 255), im=im, show=False)
    # im = plot_poly(poly_target, color=(255, 0, 0), im=im, show=False, text="Polygon IOU: {}".format(giou))
    opt.step()
    # scheduler.step()
    del loss

    opt2.zero_grad()
    l2_2 = torch.pow((mid_2_points(poly_init2) - poly_target), 2).mean()
    loss_history_2.append(c_poly_iou(mid_2_points(poly_init2), poly_target).detach().numpy())
    l2_2.backward()
    print("l2 loss: {}".format(l2_2.item()))
    # im2 = np.zeros([1000, 1000, 3]) + 255
    # im2 = plot_poly(mid_2_points(poly_init2), color=(0, 0, 255), im=im2, show=False)
    # im2 = plot_poly(poly_target, color=(255, 0, 0), im=im2, show=False, text="L2 loss: {}".format(l2_2))
    opt2.step()
    # scheduler2.step()
    del l2_2

    # Hori = np.concatenate((im, im2), axis=1)
    # cv2.imshow("im", Hori)
    # cv2.waitKey(30)

plt.title("Polygon DIoU Loss")
plt.plot(np.array(loss_history_1))
plt.show()
plt.title("L2 Loss")
plt.plot(np.array(loss_history_2))
plt.show()

print("success")