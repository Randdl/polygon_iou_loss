import numpy as np
import torch
import torch.nn as nn
import numpy as np
import cv2
from scipy.spatial import ConvexHull

def raster_poly_IOU(poly1, poly2, scale=1000):
    poly1 = (poly1 * scale / 2.0 + scale / 4.0).detach().numpy().astype(np.int32)
    poly2 = (poly2 * scale / 2.0 + scale / 4.0).detach().numpy().astype(np.int32)

    im1 = np.zeros([scale, scale])
    im2 = np.zeros([scale, scale])
    imi = np.zeros([scale, scale])

    im1 = cv2.fillPoly(im1, [poly1], color=1)
    im2 = cv2.fillPoly(im2, [poly2], color=1)
    imi = (im1 + im2) / 2.0

    cv2.imshow("imi", imi)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    imi = np.floor(imi)

    ai = np.sum(imi)
    a1 = np.sum(im1)
    a2 = np.sum(im2)

    iou = (ai) / (a1 + a2 - ai)

    return iou


def get_poly(starting_points=10):
    test = torch.rand([starting_points, 2], requires_grad=False)
    test = get_hull(test)
    return test


def get_hull(points, indices=False):
    # print(points)
    hull = ConvexHull(points.clone().cpu().detach()).vertices.astype(int)
    # print("DEBUG10")

    if indices:
        return hull

    points = points[hull, :]
    return points


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
        cv2.waitKey(1000)
        # cv2.destroyAllWindows()
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
    return im


def do_something(poly1, poly2):
    """
    Calculate the intersection over union between two convex polygons
    poly1,poly2 - [n_vertices,2] tensor of x,y,coordinates for each convex polygon
    """
    # if poly1.mean() < 1e-1:
    #     return 0
    # if poly2.mean() > 1e5:
    #     return poly2.mean()
    poly1 = torch.reshape(poly1, (4, 2))
    poly2 = torch.reshape(poly2, (4, 2))

    # print("DEBUG1")
    # poly1 = torch.zeros(4, 2).cuda().float()
    # print("DEBUG1.1")
    # poly1[0][0] = a[0]
    # print("DEBUG1.2")
    # poly1[0][1] = a[1]
    # poly1[1][0] = a[2]
    # poly1[1][1] = a[3]
    # poly1[2][0] = a[4]
    # poly1[2][1] = a[5]
    # poly1[3][0] = a[6]
    # poly1[3][1] = a[7]
    # poly2 = torch.zeros(4, 2).cuda().float()
    # poly2[0][0] = b[0]
    # poly2[0][1] = b[1]
    # poly2[1][0] = b[2]
    # poly2[1][1] = b[3]
    # poly2[2][0] = b[4]
    # poly2[2][1] = b[5]
    # poly2[3][0] = b[6]
    # poly2[3][1] = b[7]
    # print("DEBUG2")

    # blank image
    im = np.zeros([1000, 1000, 3]) + 255

    # for each polygon, sort vertices in a clockwise ordering
    poly1 = clockify(poly1)
    poly2 = clockify(poly2)

    # plot the polygons
    # im = plot_poly(poly1, color=(0, 0, 255), im=im, show=False)
    # im = plot_poly(poly2, color=(255, 0, 0), im=im, show=False)

    # find all intersection points between the two polygons - needs to be differentiable
    # we follow this formulation: https://en.wikipedia.org/wiki/Line%E2%80%93line_intersection#Given_two_points_on_each_line

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
    keep = torch.nonzero(s_total)
    # plot_poly([], im=im, show=False)

    keep = keep.cpu().detach()
    Nx = Nx[keep[:, 0], keep[:, 1]]
    Ny = Ny[keep[:, 0], keep[:, 1]]
    intersections = torch.stack([Nx, Ny], dim=1)
    # plot_poly(intersections, color=(0, 255, 0), im=im, lines=False, show=False)

    # union intersection points to poly1 and poly2 points
    union = torch.cat((poly1, poly2, intersections), dim=0)

    #  maintain an adjacency matrix
    n_elem = union.shape[0]
    p1 = poly1.shape[0]
    p2 = poly2.shape[0]

    adj1 = torch.zeros([p1, p1])
    for i in range(-1, p1 - 1):
        adj1[i, i + 1] = 1
        adj1[i + 1, i] = 1
    adj2 = torch.zeros([p2, p2])
    for i in range(-1, p2 - 1):
        adj2[i, i + 1] = 1
        adj2[i + 1, i] = 1

    adj = torch.zeros([n_elem, n_elem])
    adj[0:p1, 0:p1] = adj1
    adj[p1:p2 + p1, p1:p1 + p2] = adj2

    # plot_adjacency(union,adj,color = (0,0,0),im = None)

    # for each intersection, remove 2 connections and add 4
    for i in range(keep.shape[0]):
        xi1 = keep[i, 0]
        xi2 = (xi1 - 1) % p1
        xi3 = keep[i, 1]
        xi4 = (xi3 - 1) % p2

        xi3 = xi3.clone() + p1
        xi4 = xi4.clone() + p1

        adj[xi1, xi2] = 0
        adj[xi2, xi1] = 0
        adj[xi3, xi4] = 0
        adj[xi4, xi3] = 0

        new_idx = i + p1 + p2
        adj[new_idx, xi1] = 1
        adj[xi1, new_idx] = 1
        adj[new_idx, xi2] = 1
        adj[xi2, new_idx] = 1
        adj[new_idx, xi3] = 1
        adj[xi3, new_idx] = 1
        adj[new_idx, xi4] = 1
        adj[xi4, new_idx] = 1

        # deal with pairs of intersections on same line segment
        for j in range(keep.shape[0]):
            if i != j and (keep[j, 0] == keep[i, 0] or keep[i, 1] == keep[j, 1]):

                # connect the intersections to one another
                adj[new_idx, p1 + p2 + j] = 1
                adj[p1 + p2 + j, new_idx] = 1

                # verify that for the two endpoints of the shared segment, only one intersection is connected to each
                if keep[j, 0] == keep[i, 0]:
                    # if the x coordinate of intersection i is closer to xi1 than intersection j, adjust connections
                    if torch.abs(union[p1 + p2 + i, 0] - union[xi1,0]) < torch.abs(union[p1 + p2 + j, 0] - union[xi1,0]):  # i is closer
                        con = 1
                    else:
                        con = 0
                    adj[xi1, p1 + p2 + i] = con
                    adj[p1 + p2 + i, xi1] = con
                    adj[xi1, p1 + p2 + j] = 1 - con
                    adj[p1 + p2 + j, xi1] = 1 - con
                    adj[xi2, p1 + p2 + i] = 1 - con
                    adj[p1 + p2 + i, xi2] = 1 - con
                    adj[xi2, p1 + p2 + j] = con
                    adj[p1 + p2 + j, xi2] = con

                elif keep[j, 1] == keep[i, 1]:
                    # if the x coordinate of intersection i is closer to xi1 than intersection j, adjust connections
                    if torch.abs(union[p1 + p2 + i, 0] - union[xi3,0]) < torch.abs(union[p1 + p2 + j, 0] - union[xi3,0]):  # i is closer
                        con = 1
                    else:
                        con = 0
                    adj[xi3, p1 + p2 + i] = con
                    adj[p1 + p2 + i, xi3] = con
                    adj[xi3, p1 + p2 + j] = 1 - con
                    adj[p1 + p2 + j, xi3] = 1 - con
                    adj[xi4, p1 + p2 + i] = 1 - con
                    adj[p1 + p2 + i, xi4] = 1 - con
                    adj[xi4, p1 + p2 + j] = con
                    adj[p1 + p2 + j, xi4] = con

    # plot_adjacency(union[p1+p2:,:],adj[p1+p2:,p1+p2:],color = (0,0,0),im = im)

    # find the convex hull of the union of the polygon, and remove these points from the set of overall points
    hull_indices = get_hull(union, indices=True)
    subset_idxs = []
    for i in range(union.shape[0]):
        if i not in hull_indices:
            subset_idxs.append(i)
    subset = union[subset_idxs, :]
    adj_subset = adj[subset_idxs, :][:, subset_idxs]

    # plot the intersection
    # im = plot_poly(subset,color = (0,0,0),im = im, lines = False,show = False)
    # plot_adjacency(subset,adj_subset,color = (0,0,0),im = im)

    # repeatedly go through list and remove any points with only one collection
    changed = True
    while changed:
        changed = False
        keep_rows = torch.where(torch.sum(adj_subset, dim=0) > 1)[0]

        if len(keep_rows) < subset.shape[0]:
            changed = True

        subset = subset[keep_rows, :]
        adj_subset = adj_subset[keep_rows, :][:, keep_rows]

    # order the points in a clockwise ordering
    polyi = clockify(subset)

    # find the area of each of the convex 3 polygons - needs to be differentiable
    a1 = poly_area(poly1)
    a2 = poly_area(poly2)
    ai = poly_area(polyi)

    # print("Poly 1 area: {}".format(a1))
    # print("Poly 2 area: {}".format(a2))
    # print("Intersection area: {}".format(ai))
    iou = ai / (a1 + a2 - ai + 1e-10)
    # print("Polygon IOU: {}".format(iou))
    # plot_poly(polyi, color=(0.2, 0.7, 0.1), im=im, lines=True, text="Polygon IOU: {}".format(iou))

    return 1 - iou
    # return


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

    polygon = polygon[sorted_idxs.cpu().detach(), :]
    # print("DEBUG8")
    return polygon


def calc_iou(a, b):
    area = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])

    iw = torch.min(torch.unsqueeze(a[:, 2], dim=1), b[:, 2]) - torch.max(torch.unsqueeze(a[:, 0], 1), b[:, 0])
    ih = torch.min(torch.unsqueeze(a[:, 3], dim=1), b[:, 3]) - torch.max(torch.unsqueeze(a[:, 1], 1), b[:, 1])

    iw = torch.clamp(iw, min=0)
    ih = torch.clamp(ih, min=0)

    ua = torch.unsqueeze((a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1]), dim=1) + area - iw * ih

    ua = torch.clamp(ua, min=1e-8)

    intersection = iw * ih

    IoU = intersection / ua

    return IoU

class FocalLoss(nn.Module):
    #def __init__(self):

    def forward(self, classifications, regressions, anchors, annotations):
        top_weighting = 0.5
        alpha = 0.25
        gamma = 2.0
        batch_size = classifications.shape[0]
        classification_losses = []
        regression_losses = []
        # vp_losses = []
        
        anchor = anchors[0, :, :]

        anchor_widths  = anchor[:, 2] - anchor[:, 0]
        anchor_heights = anchor[:, 3] - anchor[:, 1]
        anchor_ctr_x   = anchor[:, 0] + 0.5 * anchor_widths
        anchor_ctr_y   = anchor[:, 1] + 0.5 * anchor_heights

        # separate vp terms
        # vps = annotations[:,0,21:27] # should be [b,6]
        # annotations = annotations[:,:,:21]

        for j in range(batch_size):

            classification = classifications[j, :, :]
            regression = regressions[j, :, :]
            # vp = vps[j]
            
            bbox_annotation = annotations[j, :, :]
            bbox_annotation = bbox_annotation[bbox_annotation[:, -1] != -1]
            
            classification = torch.clamp(classification, 1e-4, 1.0 - 1e-4)

            if bbox_annotation.shape[0] == 0:
                if torch.cuda.is_available():
                    alpha_factor = torch.ones(classification.shape).cuda() * alpha

                    alpha_factor = 1. - alpha_factor
                    focal_weight = classification
                    focal_weight = alpha_factor * torch.pow(focal_weight, gamma)

                    bce = -(torch.log(1.0 - classification))

                    # cls_loss = focal_weight * torch.pow(bce, gamma)
                    cls_loss = focal_weight * bce
                    classification_losses.append(cls_loss.sum())
                    regression_losses.append(torch.tensor(0).float())
                    
                else:
                    alpha_factor = torch.ones(classification.shape) * alpha

                    alpha_factor = 1. - alpha_factor
                    focal_weight = classification
                    focal_weight = alpha_factor * torch.pow(focal_weight, gamma)

                    bce = -(torch.log(1.0 - classification))

                    # cls_loss = focal_weight * torch.pow(bce, gamma)
                    cls_loss = focal_weight * bce
                    classification_losses.append(cls_loss.sum())
                    regression_losses.append(torch.tensor(0).float())
                    
                continue

            # calculate the bounding shell of the polygon
            
            xmin,_ = torch.min(bbox_annotation[:,[0,2,4,6]],dim = 1)
            xmax,_ = torch.max(bbox_annotation[:,[0,2,4,6]],dim = 1)
            ymin,_ = torch.min(bbox_annotation[:,[1,3,5,7]],dim = 1)
            ymax,_ = torch.max(bbox_annotation[:,[1,3,5,7]],dim = 1)

            # xmin2,_ = torch.min(bbox_annotation[:,[8,10,12,14]],dim = 1)
            # xmax2,_ = torch.max(bbox_annotation[:,[8,10,12,14]],dim = 1)
            # ymin2,_ = torch.min(bbox_annotation[:,[9,11,13,15]],dim = 1)
            # ymax2,_ = torch.max(bbox_annotation[:,[9,11,13,15]],dim = 1)
            
            xmin = xmin.unsqueeze(1)
            xmax = xmax.unsqueeze(1)
            ymin = ymin.unsqueeze(1)
            ymax = ymax.unsqueeze(1)
            bbox_annotation_2D = torch.cat((xmin,ymin,xmax,ymax),dim = 1)
            
            IoU = calc_iou(anchors[0, :, :], bbox_annotation_2D) # num_anchors x num_annotations
            IoU_max, IoU_argmax = torch.max(IoU, dim=1) # num_anchors x 1

            #import pdb
            #pdb.set_trace()

            # compute the loss for classification
            targets = torch.ones(classification.shape) * -1

            if torch.cuda.is_available():
                targets = targets.cuda()

            targets[torch.lt(IoU_max, 0.4), :] = 0
            

            positive_indices = torch.ge(IoU_max, 0.5)

            num_positive_anchors = positive_indices.sum()

            assigned_annotations = bbox_annotation[IoU_argmax, :]

            targets[positive_indices, :] = 0
            targets[positive_indices, assigned_annotations[positive_indices, -1].long()] = 1

            if torch.cuda.is_available():
                alpha_factor = torch.ones(targets.shape).cuda() * alpha
            else:
                alpha_factor = torch.ones(targets.shape) * alpha

            alpha_factor = torch.where(torch.eq(targets, 1.), alpha_factor, 1. - alpha_factor)
            focal_weight = torch.where(torch.eq(targets, 1.), 1. - classification, classification)
            focal_weight = alpha_factor * torch.pow(focal_weight, gamma)

            bce = -(targets * torch.log(classification) + (1.0 - targets) * torch.log(1.0 - classification))

            # cls_loss = focal_weight * torch.pow(bce, gamma)
            cls_loss = focal_weight * bce

            if torch.cuda.is_available():
                cls_loss = torch.where(torch.ne(targets, -1.0), cls_loss, torch.zeros(cls_loss.shape).cuda())
            else:
                cls_loss = torch.where(torch.ne(targets, -1.0), cls_loss, torch.zeros(cls_loss.shape))

            classification_losses.append(cls_loss.sum()/torch.clamp(num_positive_anchors.float(), min=1.0))

            # compute the loss for regression

            if positive_indices.sum() > 0:
                assigned_annotations = assigned_annotations[positive_indices, :]
                anchor_widths_pi = anchor_widths[positive_indices]
                anchor_heights_pi = anchor_heights[positive_indices]
                anchor_ctr_x_pi = anchor_ctr_x[positive_indices]
                anchor_ctr_y_pi = anchor_ctr_y[positive_indices]


                ### HERE, we'll need to redefine this logic for 3D bbox formulation
                # normalize coordinates by width and height
                # normalize tail length by dividing then taking log
                
                # gt_widths  = assigned_annotations[:, 2] - assigned_annotations[:, 0]
                # gt_heights = assigned_annotations[:, 3] - assigned_annotations[:, 1]
                # gt_ctr_x   = assigned_annotations[:, 0] + 0.5 * gt_widths
                # gt_ctr_y   = assigned_annotations[:, 1] + 0.5 * gt_heights

                # # clip widths to 1
                # gt_widths  = torch.clamp(gt_widths, min=1)
                # gt_heights = torch.clamp(gt_heights, min=1)

                # targets_dx = (gt_ctr_x - anchor_ctr_x_pi) / anchor_widths_pi
                # targets_dy = (gt_ctr_y - anchor_ctr_y_pi) / anchor_heights_pi
                # targets_dw = torch.log(gt_widths / anchor_widths_pi)
                # targets_dh = torch.log(gt_heights / anchor_heights_pi)
                
                # targets = torch.stack((targets_dx, targets_dy, targets_dw, targets_dh))
                # targets = targets.t()
                targets = assigned_annotations[:,:-1] # remove classifications from targets
                
                # regression is x,y,lx,ly,wx,wy,hx,hy - need to convert to corner coordinates
                # fbl fbr bbl bbr ftl ftr btl btr - alternate w first, then l, then h
            
                regression = regression[positive_indices,:] #[n_positive_indices, 12]
                # print(targets.shape)
                # print(regression.shape)
                
                # # vp = vp.unsqueeze(0).repeat(len(regression),1)
                # # # expand  anchor dims
                # # vp_anchor_widths    = anchor_widths_pi.unsqueeze(1).repeat(1,3)
                # # vp_anchor_heights   = anchor_heights_pi.unsqueeze(1).repeat(1,3)
                # # vp_anchor_x         = anchor_ctr_x_pi.unsqueeze(1).repeat(1,3)
                # # vp_anchor_y         = anchor_ctr_y_pi.unsqueeze(1).repeat(1,3)
                #
                # # # we'll need to convert the vanishing points first
                # # vp[:,[0,2,4]] = (vp[:,[0,2,4]] -vp_anchor_x ) / vp_anchor_widths
                # # vp[:,[1,3,5]] = (vp[:,[1,3,5]] -vp_anchor_y ) / vp_anchor_heights
                #
                # # regression has shape [b,a,12]
                # # obj_ctr_x = regression[:,0]
                # # obj_ctr_y = regression[:,1]
                #
                # # VP vectors are computed from object center to vanishing point
                # # Object vectors are computed towards the back, towards the right, and towards the top
                # # Thus, if the vanishing point is closer to the back/right/top, we want the angle to be 0 (cos term = 1)
                # # Otherwise, we want the angle to be 180 (cos term = 1)
                #
                # # we can look at the assigned annotation for each prediction, and compute a scale factor (1 or -1) based on the box's orientation
                # # then multiply the cos terms by this factor at the end
                #
                # ### VP 1
                # # we compute the line from each box towards each vp direction
                # #vector components
                # reg_vec_x = regression[:,2]
                # reg_vec_y = regression[:,3]
                #
                # # vector is in direction front -> back
                # targ_vec_x = ((targets[:,4] + targets[:,6] + targets[:,12] + targets[:,14]) - (targets[:,0] + targets[:,2] + targets[:,8] + targets[:,10]) )/4.0
                # targ_vec_y = ((targets[:,5] + targets[:,7] + targets[:,13] + targets[:,15]) - (targets[:,1] + targets[:,3] + targets[:,9] + targets[:,11]) )/4.0
                #
                # # dot product
                # reg_norm = torch.sqrt(torch.pow(reg_vec_x,2) + torch.pow(reg_vec_y,2))
                # targ_norm = torch.sqrt(torch.pow(targ_vec_x,2) + torch.pow(targ_vec_y,2))
                # cos_angle = (reg_vec_x * targ_vec_x + reg_vec_y * targ_vec_y)/(reg_norm * targ_norm)
                #
                # ## based on new definition, don't need the sign term
                # # # sign term
                # # # distance of front bottom left from vp
                # # d1 = torch.sqrt((vp[:,0] - targets[:,0])**2 + (vp[:,1] - targets[:,1])**2)
                # # # distance of back bottom left from vp
                # # d2 = torch.sqrt((vp[:,0] - targets[:,4])**2 + (vp[:,1] - targets[:,5])**2)
                # # # if back is closer than front (d1-d2) + , reg_vec points towards vp (sign +)
                # # sign_vec = torch.sign(d1-d2)
                # # multiply loss term by sign
                #
                # vp1_loss = 1- cos_angle
                #
                # # for our loss term we'll use 1-cos(angle) = 1- vec1 . vec2 / (||vec1||*||vec2||)
                # # we have to consider both reg vector orientations and take best
                # #vp1_loss = 1-torch.pow(cos_angle,2)
                #
                # ### VP 2
                # # we compute the line from each box towards each vp direction
                # # vector components
                # reg_vec_x = regression[:,4]
                # reg_vec_y = regression[:,5]
                #
                # # vector is in direction left -> right (so add right side, then subtract left side)
                # targ_vec_x = ((targets[:,2] + targets[:,6] + targets[:,10] + targets[:,14]) - (targets[:,0] + targets[:,4] + targets[:,8] + targets[:,12]) )/4.0
                # targ_vec_y = ((targets[:,3] + targets[:,7] + targets[:,11] + targets[:,15]) - (targets[:,1] + targets[:,5] + targets[:,9] + targets[:,13]) )/4.0
                #
                # # dot product
                # reg_norm = torch.sqrt(torch.pow(reg_vec_x,2) + torch.pow(reg_vec_y,2))
                # targ_norm = torch.sqrt(torch.pow(targ_vec_x,2) + torch.pow(targ_vec_y,2))
                # cos_angle = (reg_vec_x * targ_vec_x + reg_vec_y * targ_vec_y)/(reg_norm * targ_norm)
                #
                # # # sign term
                # # # distance of front bottom left from vp
                # # d1 = torch.sqrt((vp[:,2] - targets[:,0])**2 + (vp[:,3] - targets[:,1])**2)
                # # # distance of front bottom right from vp
                # # d2 = torch.sqrt((vp[:,2] - targets[:,2])**2 + (vp[:,3] - targets[:,3])**2)
                # # # sign should be positive if right is closer
                # # sign_vec = torch.sign(d1-d2)
                # # # multiply loss term by sign
                # vp2_loss = 1- cos_angle
                #
                # # for our loss term we'll use 1-cos(angle) = 1- vec1 . vec2 / (||vec1||*||vec2||)
                # # we have to consider both reg vector orientations and take best
                # #vp2_loss = 1-torch.pow(cos_angle,2)
                #
                # ### VP 3
                # # we compute the line from each box towards each vp direction
                # # vector_components
                # reg_vec_x = regression[:,6]
                # reg_vec_y = regression[:,7]
                #
                # # vector is in direction top -> bottom (so add bottom, then subtract top)
                # targ_vec_x = ((targets[:,0] + targets[:,2] + targets[:,4] + targets[:,6]) - (targets[:,8] + targets[:,10] + targets[:,12] + targets[:,14]) )/4.0
                # targ_vec_y = ((targets[:,1] + targets[:,3] + targets[:,5] + targets[:,7]) - (targets[:,9] + targets[:,11] + targets[:,13] + targets[:,15]) )/4.0
                #
                # # dot product
                # reg_norm = torch.sqrt(torch.pow(reg_vec_x,2) + torch.pow(reg_vec_y,2))
                # targ_norm = torch.sqrt(torch.pow(targ_vec_x,2) + torch.pow(targ_vec_y,2))
                # cos_angle = (reg_vec_x * targ_vec_x + reg_vec_y * targ_vec_y)/(reg_norm * targ_norm)
                #
                # # # sign term
                # # # distance of front bottom left from vp
                # # d1 = torch.sqrt((vp[:,4] - targets[:,0])**2 + (vp[:,5] - targets[:,1])**2)
                # # # distance of front top left from vp
                # # d2 = torch.sqrt((vp[:,4] - targets[:,8])**2 + (vp[:,5] - targets[:,9])**2)
                # # # if bottom is closer (d2-d1) +, sign should be +
                # # sign_vec = torch.sign(d2-d1)
                # # # multiply loss term by sign
                # vp3_loss = 1- cos_angle
                #
                # # for our loss term we'll use 1-cos(angle) = 1- vec1 . vec2 / (||vec1||*||vec2||)
                # # we have to consider both reg vector orientations and take best
                # #vp3_loss = 1-torch.pow(cos_angle,2)
                #
                # vp_loss = (vp1_loss + vp2_loss + vp3_loss)/3.0
                # vp_losses.append(vp_loss.mean())
                #
                #
                # # try to introduce bias so all directions are equally possible anglewise /???
                # #regression[:,2:] -= 0.5
                #
                # preds = torch.zeros([regression.shape[0],20],requires_grad = True).cuda()
                # preds[:,0] = regression[:,0] - regression[:,2] - regression[:,4] + regression[:,6]
                # preds[:,1] = regression[:,1] - regression[:,3] - regression[:,5] + regression[:,7]
                # preds[:,2] = regression[:,0] - regression[:,2] + regression[:,4] + regression[:,6]
                # preds[:,3] = regression[:,1] - regression[:,3] + regression[:,5] + regression[:,7]
                # preds[:,4] = regression[:,0] + regression[:,2] - regression[:,4] + regression[:,6]
                # preds[:,5] = regression[:,1] + regression[:,3] - regression[:,5] + regression[:,7]
                # preds[:,6] = regression[:,0] + regression[:,2] + regression[:,4] + regression[:,6]
                # preds[:,7] = regression[:,1] + regression[:,3] + regression[:,5] + regression[:,7]
                #
                # preds[:,8]  = regression[:,0] - regression[:,2] - regression[:,4] - regression[:,6]
                # preds[:,9]  = regression[:,1] - regression[:,3] - regression[:,5] - regression[:,7]
                # preds[:,10] = regression[:,0] - regression[:,2] + regression[:,4] - regression[:,6]
                # preds[:,11] = regression[:,1] - regression[:,3] + regression[:,5] - regression[:,7]
                # preds[:,12] = regression[:,0] + regression[:,2] - regression[:,4] - regression[:,6]
                # preds[:,13] = regression[:,1] + regression[:,3] - regression[:,5] - regression[:,7]
                # preds[:,14] = regression[:,0] + regression[:,2] + regression[:,4] - regression[:,6]
                # preds[:,15] = regression[:,1] + regression[:,3] + regression[:,5] - regression[:,7]
                # preds[:,16:20] = regression[:,8:12]
                #
                # targets[:,[0,2,4,6,8,10,12,14,16,18]] = (targets[:,[0,2,4,6,8,10,12,14,16,18]] - anchor_ctr_x_pi.unsqueeze(1).repeat(1,10)) / anchor_widths_pi.unsqueeze(1).repeat(1,10)
                # targets[:,[1,3,5,7,9,11,13,15,17,19]] = (targets[:,[1,3,5,7,9,11,13,15,17,19]] - anchor_ctr_y_pi.unsqueeze(1).repeat(1,10)) / anchor_heights_pi.unsqueeze(1).repeat(1,10)
                #
                # std_dev
                # targets = targets/(0.1*torch.ones([8]).cuda())
                #
                #
                # negative_indices = 1 + (~positive_indices)

                # regression_diff = torch.abs(targets - regression[positive_indices, :])
                # regression_diff = torch.abs(targets - regression)
                # regression_diff = torch.square(torch.abs(targets - regression))
                # regression_losses.append(torch.sqrt(regression_diff.mean()) / 100)
                for ii in range(targets.shape[0]):
                    regression_losses.append(do_something(targets[ii], regression[ii]))

                # here, we underweight the top corner coords by a factor of self.top_weighting
                # regression_diff[:,8:16] *= top_weighting
                
                # regression_loss = torch.where(
                #     torch.le(regression_diff, 1.0 / 9.0),
                #     0.5 * 9.0 * torch.pow(regression_diff, 2),
                #     regression_diff - 0.5 / 9.0
                # )

            else:
                if torch.cuda.is_available():
                    # vp_losses.append(torch.tensor(0).float().cuda())
                    regression_losses.append(torch.tensor(0).float().cuda())
                else:
                    # vp_losses.append(torch.tensor(0).float())
                    regression_losses.append(torch.tensor(0).float())
                    
        classification_losses = [item.cuda() for item in classification_losses]
        regression_losses = [item.cuda() for item in regression_losses]
        # vp_losses = [item.cuda() for item in vp_losses]
        return torch.stack(classification_losses).mean(dim=0, keepdim=True), torch.stack(regression_losses).mean(dim=0, keepdim=True)

    
