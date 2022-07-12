import torch
import numpy as np
import cv2


def poly_area(polygon):
    """
    Returns the area of the polygon
    polygon - [n_vertices,2] tensor of clockwise points
    """
    x1 = polygon[:, 0]
    y1 = polygon[:, 1]

    x2 = x1.roll(1)
    y2 = y1.roll(1)

    # per this formula: http://www.mathwords.com/a/area_convex_polygon.htm
    area = -1 / 2.0 * (torch.sum(x1 * y2) - torch.sum(x2 * y1))

    return area


def torch_wn(pnts, poly, return_winding=False):
    # print(poly)
    x0, y0 = poly[:].T  # polygon `from` coordinates
    # print(x0)
    x1, y1 = torch.roll(poly, -1, 0).T  # polygon `to` coordinates
    # print(x1)
    x, y = pnts.T  # point coordinates
    y_y0 = y[:, None] - y0
    x_x0 = x[:, None] - x0
    diff_ = (x1 - x0) * y_y0 - (y1 - y0) * x_x0  # diff => einsum in original
    chk1 = (y_y0 > 0.0)
    chk2 = torch.less(y[:, None], y1)  # pnts[:, 1][:, None], poly[1:, 1])
    chk3 = torch.sign(diff_)
    pos = (chk1 & chk2 & (chk3 > 0)).sum(axis=1, dtype=int)
    neg = (~chk1 & ~chk2 & (chk3 < 0)).sum(axis=1, dtype=int)
    wn = pos - neg
    out_ = pnts[torch.nonzero(wn)[:, 0]]
    if return_winding:
        return out_, wn
    return out_


def c_poly_iou(poly1, poly2):
    """
    Calculate the intersection over union between two convex polygons
    poly1,poly2 - [n_vertices,2] tensor of x,y,coordinates for each convex polygon
    """
    poly1 = clockify(poly1)
    poly2 = clockify(poly2)
    # print(poly1)
    # print(poly2)

    # im = np.zeros([1000, 1000, 3]) + 255
    # im = plot_poly(poly1, color=(0, 0, 255), im=im, show=False)
    # im = plot_poly(poly2, color=(255, 0, 0), im=im, show=False)
    # plot_poly([], im=im, show=False)

    # tensors such that elementwise each index corresponds to the interstection of a poly1 line and poly2 line
    xy1 = poly1.unsqueeze(1).expand([-1, poly2.shape[0], -1])
    xy3 = poly2.unsqueeze(0).expand([poly1.shape[0], -1, -1])

    # same data, but rolled to next element
    xy2 = poly1.roll(1, 0).unsqueeze(1).expand([-1, poly2.shape[0], -1])
    xy4 = poly2.roll(1, 0).unsqueeze(0).expand([poly1.shape[0], -1, -1])

    x1 = xy1[:, :, 0]
    y1 = xy1[:, :, 1]
    x2 = xy2[:, :, 0]
    y2 = xy2[:, :, 1]
    x3 = xy3[:, :, 0]
    y3 = xy3[:, :, 1]
    x4 = xy4[:, :, 0]
    y4 = xy4[:, :, 1]

    # Nx and Ny contain x and y intersection coordinates for each pair of line segments
    D = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
    Nx = ((x1 * y2 - x2 * y1) * (x3 - x4) - (x3 * y4 - x4 * y3) * (x1 - x2)) / D
    Ny = ((x1 * y2 - x2 * y1) * (y3 - y4) - (x3 * y4 - x4 * y3) * (y1 - y2)) / D

    # get points that intersect in valid range (Nx should be greater than exactly one of x1,x2 and exactly one of x3,x4)
    s1 = torch.sign(Nx - x1)
    s2 = torch.sign(Nx - x2)
    s12 = (s1 * s2 - 1) / -2.0
    s3 = torch.sign(Nx - x3)
    s4 = torch.sign(Nx - x4)
    s34 = (s3 * s4 - 1) / -2.0
    s_total = s12 * s34  # 1 if line segments intersect, 0 otherwise
    # modified
    # s_total[D < 1e-10] = 0
    keep = torch.nonzero(s_total)

    keep = keep.detach()
    Nx = Nx[keep[:, 0], keep[:, 1]]
    Ny = Ny[keep[:, 0], keep[:, 1]]
    intersections = torch.stack([Nx, Ny], dim=1)
    # print(np_wn(poly1_np, poly2_np))
    poly1_np_keep = torch_wn(poly1, poly2)
    poly2_np_keep = torch_wn(poly2, poly1)
    # poly1_np_keep = torch.tensor(poly1_np_keep)
    # poly2_np_keep = torch.tensor(poly2_np_keep)

    # plot_poly(intersections, color=(0, 255, 0), im=im, lines=False, show=True)

    union = torch.cat((poly1_np_keep, poly2_np_keep, intersections), dim=0)
    # print(intersections)
    polyi = clockify(union)
    # print(polyi)

    a1 = poly_area(poly1)
    a2 = poly_area(poly2)
    ai = poly_area(polyi)

    # print("Poly 1 area: {}".format(a1))
    # print("Poly 2 area: {}".format(a2))
    # print("Intersection area: {}".format(ai))
    iou = ai / (a1 + a2 - ai + 1e-10)

    # plot_poly(polyi, color=(0, 0, 0), im=im, lines=True, text="Polygon IOU: {}".format(iou))

    return iou


def c_poly_loss(poly1, poly2):
    return 1 - c_poly_iou(poly1, poly2)


def c_poly_diou_loss(poly1, poly2):
    iou = c_poly_iou(poly1, poly2)

    # find the smallest 2d box that contains both polygons
    x_min = min(torch.min(poly1[:, 0]), torch.min(poly2[:, 0]))
    x_max = max(torch.max(poly1[:, 0]), torch.max(poly2[:, 0]))
    y_min = min(torch.min(poly1[:, 1]), torch.min(poly2[:, 1]))
    y_max = max(torch.max(poly1[:, 1]), torch.max(poly2[:, 1]))
    # the diagonal length of the smallest enclosing box covering two polygons
    c_sqd = torch.square(x_max - x_min) + torch.square(y_max - y_min)

    poly_1_mid = torch.mean(poly1, dim=0)
    poly_2_mid = torch.mean(poly2, dim=0)
    # the euclidian distance between the center point of two polygons
    d_sqd = torch.sum(torch.square(poly_1_mid - poly_2_mid))

    return 1 - iou + d_sqd / c_sqd


def c_poly_giou(poly1, poly2):
    """
    Calculate the giou between two convex polygons
    poly1,poly2 - [n_vertices,2] tensor of x,y,coordinates for each convex polygon
    """
    poly1 = clockify(poly1)
    poly2 = clockify(poly2)
    # print(poly1)
    # print(poly2)

    # im = np.zeros([1000, 1000, 3]) + 255
    # im = plot_poly(poly1, color=(0, 0, 255), im=im, show=False)
    # im = plot_poly(poly2, color=(255, 0, 0), im=im, show=False)
    # plot_poly([], im=im, show=False)

    # tensors such that elementwise each index corresponds to the interstection of a poly1 line and poly2 line
    xy1 = poly1.unsqueeze(1).expand([-1, poly2.shape[0], -1])
    xy3 = poly2.unsqueeze(0).expand([poly1.shape[0], -1, -1])

    # same data, but rolled to next element
    xy2 = poly1.roll(1, 0).unsqueeze(1).expand([-1, poly2.shape[0], -1])
    xy4 = poly2.roll(1, 0).unsqueeze(0).expand([poly1.shape[0], -1, -1])

    x1 = xy1[:, :, 0]
    y1 = xy1[:, :, 1]
    x2 = xy2[:, :, 0]
    y2 = xy2[:, :, 1]
    x3 = xy3[:, :, 0]
    y3 = xy3[:, :, 1]
    x4 = xy4[:, :, 0]
    y4 = xy4[:, :, 1]

    # Nx and Ny contain x and y intersection coordinates for each pair of line segments
    D = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
    Nx = ((x1 * y2 - x2 * y1) * (x3 - x4) - (x3 * y4 - x4 * y3) * (x1 - x2)) / D
    Ny = ((x1 * y2 - x2 * y1) * (y3 - y4) - (x3 * y4 - x4 * y3) * (y1 - y2)) / D

    # get points that intersect in valid range (Nx should be greater than exactly one of x1,x2 and exactly one of x3,x4)
    s1 = torch.sign(Nx - x1)
    s2 = torch.sign(Nx - x2)
    s12 = (s1 * s2 - 1) / -2.0
    s3 = torch.sign(Nx - x3)
    s4 = torch.sign(Nx - x4)
    s34 = (s3 * s4 - 1) / -2.0
    s_total = s12 * s34  # 1 if line segments intersect, 0 otherwise
    # modified
    # s_total[D < 1e-10] = 0
    keep = torch.nonzero(s_total)

    keep = keep.detach()
    Nx = Nx[keep[:, 0], keep[:, 1]]
    Ny = Ny[keep[:, 0], keep[:, 1]]
    intersections = torch.stack([Nx, Ny], dim=1)
    # print(np_wn(poly1_np, poly2_np))
    poly1_np_keep = torch_wn(poly1, poly2)
    poly2_np_keep = torch_wn(poly2, poly1)
    # poly1_np_keep = torch.tensor(poly1_np_keep)
    # poly2_np_keep = torch.tensor(poly2_np_keep)

    # plot_poly(intersections, color=(0, 255, 0), im=im, lines=False, show=True)

    union = torch.cat((poly1_np_keep, poly2_np_keep, intersections), dim=0)
    # print(intersections)
    polyi = clockify(union)
    # print(polyi)

    # find the smallest 2d box that contains both polygons
    x_min = min(torch.min(poly1[:, 0]), torch.min(poly2[:, 0]))
    x_max = max(torch.max(poly1[:, 0]), torch.max(poly2[:, 0]))
    y_min = min(torch.min(poly1[:, 1]), torch.min(poly2[:, 1]))
    y_max = max(torch.max(poly1[:, 1]), torch.max(poly2[:, 1]))
    ag = (x_max - x_min) * (y_max - y_min)

    # find the area of intersection
    ai = poly_area(polyi)

    # print("Poly 1 area: {}".format(a1))
    # print("Poly 2 area: {}".format(a2))
    # print("Intersection area: {}".format(ai))
    giou = ai / (ag + 1e-10)

    # plot_poly(polyi, color=(0, 0, 0), im=im, lines=True, text="Polygon IOU: {}".format(iou))

    return giou


def clockify(polygon, clockwise=True):
    """
    polygon - [n_vertices,2] tensor of x,y,coordinates for each convex polygon
    clockwise - if True, clockwise, otherwise counterclockwise
    returns - [n_vertices,2] tensor of sorted coordinates
    """

    # get center
    center = torch.mean(polygon, dim=0)

    # get angle to each point from center
    diff = polygon - center.unsqueeze(0).expand([polygon.shape[0], 2])
    tan = torch.atan(diff[:, 1] / diff[:, 0])
    direction = (torch.sign(diff[:, 0]) - 1) / 2.0 * -np.pi

    angle = tan + direction
    sorted_idxs = torch.argsort(angle)

    if not clockwise:
        sorted_idxs.reverse()

    polygon = polygon[sorted_idxs.detach(), :]
    return polygon


def mid_2_points(mids):
    mid = torch.unsqueeze(mids[0, :], 0)
    first_quadrant = torch.unsqueeze(mids[1, :], 0)
    second_quadrant = torch.unsqueeze(mids[2, :], 0)
    second_quadrant[0, 0].add(-mids[2, 0])
    return torch.cat((mid+first_quadrant, mid-second_quadrant, mid-first_quadrant, mid+second_quadrant), dim=0)
