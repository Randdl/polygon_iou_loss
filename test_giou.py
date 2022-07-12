import torch
import cv2
from polyogn_iou_loss import *

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

init_mid = [[0.4077, 0.5054],
         [0.2052, 0.3274],
         [0.2305, 0.3237]]
init_mid = torch.Tensor(init_mid)
init_mid2 = [[0.4077, 0.5054],
         [0.2052, 0.3274],
         [0.2305, 0.3237]]
init_mid2 = torch.Tensor(init_mid2)
target_mid = [[0.3077, 0.6054],
         [0.1052, 0.3274],
         [0.2305, 0.1837]]
target_mid = torch.Tensor(target_mid)
# print(mid_2_points(init_mid))

# poly1 = get_poly(starting_points=4) * torch.rand(1) + torch.rand(1)
# poly1 = [[0.9077, 0.8054],
#          [0.8052, 0.9274],
#          [0.7305, 0.9237],
#          [0.7336, 0.7877]]
# poly1 = torch.Tensor(poly1)
# print(poly1.shape)
#
# poly2 = get_poly(starting_points=4)
# poly2 = [[0.8070, 0.4038],
#          [0.9020, 0.9710],
#          [0.8386, 0.9664],
#          [0.5124, 0.6275]]
# poly2 = torch.Tensor(poly2)
# print(poly2.shape)

# poly = torch.autograd.Variable(poly1, requires_grad=True)
poly_init = torch.autograd.Variable(init_mid, requires_grad=True)
poly_init2 = torch.autograd.Variable(init_mid2, requires_grad=True)
poly_target = mid_2_points(target_mid)

opt = torch.optim.Adam([poly_init], lr=0.015)
scheduler = torch.optim.lr_scheduler.StepLR(opt, step_size=30, gamma=0.1)
opt2 = torch.optim.Adam([poly_init2], lr=0.015)
scheduler2 = torch.optim.lr_scheduler.StepLR(opt2, step_size=30, gamma=0.1)

giou = c_poly_giou(mid_2_points(poly_init), poly_target)

for k in range(1000):
    giou = c_poly_giou(mid_2_points(poly_init), poly_target)
    opt.zero_grad()
    lp = (1.0 - giou)
    l2 = torch.pow((mid_2_points(poly_init) - poly_target), 2).mean()
    loss = c_poly_diou_loss(mid_2_points(poly_init), poly_target)
    loss.backward()
    print("Polygon IOU: {}. lr:{}".format(giou.item(), opt.param_groups[0]['lr']))
    im = np.zeros([1000, 1000, 3]) + 255
    im = plot_poly(mid_2_points(poly_init), color=(0, 0, 255), im=im, show=False)
    im = plot_poly(poly_target, color=(255, 0, 0), im=im, show=False, text="Polygon IOU: {}".format(giou))
    opt.step()
    scheduler.step()
    del loss

    opt2.zero_grad()
    l2_2 = torch.pow((mid_2_points(poly_init2) - poly_target), 2).mean()
    l2_2.backward()
    print("l2 loss: {}".format(l2_2.item()))
    im2 = np.zeros([1000, 1000, 3]) + 255
    im2 = plot_poly(mid_2_points(poly_init2), color=(0, 0, 255), im=im2, show=False)
    im2 = plot_poly(poly_target, color=(255, 0, 0), im=im2, show=False, text="L2 loss: {}".format(l2_2))
    opt2.step()
    scheduler2.step()
    del l2_2

    Hori = np.concatenate((im, im2), axis=1)
    cv2.imshow("im", Hori)
    cv2.waitKey(300)
    # cv2.destroyAllWindows()

print("success")