import torch
import numpy as np
import cv2

from matplotlib import pyplot as plt

device = 'cuda'


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
	x0, y0 = poly[:].T  # polygon `from` coordinates
	x1, y1 = torch.roll(poly, -1, 0).T  # polygon `to` coordinates

	x, y = pnts.T  # point coordinates

	y_y0 = y[:, None] - y0
	x_x0 = x[:, None] - x0
	diff_ = (x1 - x0) * y_y0 - (y1 - y0) * x_x0  # diff => einsum in original
	chk1 = (y_y0 > 0.0)
	chk2 = torch.lt(y[:, None], y1)  # pnts[:, 1][:, None], poly[1:, 1])
	chk3 = torch.sign(diff_)
	# print('chk1 is:\n', chk1)
	# print('chk2 is:\n', chk2)
	# print('chk3 is:\n', chk3)

	pos = (chk1 & chk2 & (chk3 > 0)).sum(axis=1, dtype=int)
	neg = (~chk1 & ~chk2 & (chk3 < 0)).sum(axis=1, dtype=int)
	wn = pos - neg
	# print(wn)
	# print(torch.nonzero(wn))
	# print('with pnts', pnts)
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
	Nx = ((x1 * y2 - x2 * y1) * (x3 - x4) - (x3 * y4 - x4 * y3) * (x1 - x2)) / (D + 1e-10)
	Ny = ((x1 * y2 - x2 * y1) * (y3 - y4) - (x3 * y4 - x4 * y3) * (y1 - y2)) / (D + 1e-10)

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


def batch_poly_area(polys):
	"""
	Calculate the area of polygons.
	:param polys: the corner points of polygons [B, N, 2]
	:return: area of the polygon [B]
	"""
	x1 = polys[:, :, 0]
	y1 = polys[:, :, 1]

	x2 = x1.roll(1, 1)
	y2 = y1.roll(1, 1)

	# per this formula: http://www.mathwords.com/a/area_convex_polygon.htm
	area = -1 / 2.0 * (torch.sum(x1 * y2, dim=1) - torch.sum(x2 * y1, dim=1))

	return area


def batch_clockify(polygons, clockwise=True):
	"""
	Turn the points of the polygons into clockwise order
	:param polygons: the corner points of polygons [B, N, 2]
	:return: [B, N, 2]
	"""
	center = torch.mean(polygons, dim=1)

	diff = polygons - center.unsqueeze(1).expand(polygons.shape)
	tan = torch.atan(diff[:, :, 1] / diff[:, :, 0])
	direction = (torch.sign(diff[:, :, 0]) - 1) / 2.0 * -np.pi

	angle = tan + direction

	sorted_idxs = torch.argsort(angle)

	if not clockwise:
		sorted_idxs.reverse()

	first_indices = torch.arange(sorted_idxs.shape[0])[:, None]
	polygons = polygons[first_indices, sorted_idxs.detach()]

	return polygons


def batch_torch_wn(pntss, polys, return_winding=False, sides=4):
	"""
	Return the points inside polygons
	:param pntss: The set of points to check whether inside polygons [B, N, 2]
	:param polys: [B, N, 2]
	:param return_winding: whether return the winding value
	:param sides: number of sides of the polygons
	:return: [B, N', 2]
	"""
	device = polys.device
	x0, y0 = torch.transpose(polys, 0, 2).transpose(-2, -1)  # polygon `from` coordinates
	x1, y1 = torch.transpose(torch.roll(polys, -1, 1), 0, 2).transpose(-2, -1)  # polygon `to` coordinates

	x, y = torch.transpose(pntss, 0, 2).transpose(-2, -1)  # point coordinates

	y_y0 = y[:, :, None] - y0[:, None]
	x_x0 = x[:, :, None] - x0[:, None]
	diff_ = (x1 - x0)[:, None] * y_y0[:, :] - (y1 - y0)[:, None] * x_x0  # diff => einsum in original
	chk1 = (y_y0 > 0.0)
	chk2 = torch.lt(y[:, :, None], y1[:, None])  # pnts[:, 1][:, None], poly[1:, 1])
	chk3 = torch.sign(diff_)
	# print('chk1 is:\n', chk1)
	# print('chk2 is:\n', chk2)
	# print('chk3 is:\n', chk3)

	pos = (chk1 & chk2 & (chk3 > 0)).sum(axis=2, dtype=int)
	neg = (~chk1 & ~chk2 & (chk3 < 0)).sum(axis=2, dtype=int)
	wn = pos - neg
	# print(wn)
	# print(torch.nonzero(wn))
	# print('with pntss:', pntss)
	out_ = torch.zeros((pntss.shape[0], sides, 2)).to(device)
	idxs = torch.where(wn != 0)
	out_[idxs] = pntss[idxs]

	# out_ = pntss[torch.nonzero(wn)[:, 0]]
	if return_winding:
		return out_, wn
	return out_


def batch_torch_wn_triangle(pntss, polys, return_winding=False):
	device = polys.device
	x0, y0 = torch.transpose(polys, 0, 2).transpose(-2, -1)  # polygon `from` coordinates
	x1, y1 = torch.transpose(torch.roll(polys, -1, 1), 0, 2).transpose(-2, -1)  # polygon `to` coordinates

	x, y = torch.transpose(pntss, 0, 2).transpose(-2, -1)  # point coordinates

	y_y0 = y[:, :, None] - y0[:, None]
	x_x0 = x[:, :, None] - x0[:, None]
	diff_ = (x1 - x0)[:, None] * y_y0[:, :] - (y1 - y0)[:, None] * x_x0  # diff => einsum in original
	chk1 = (y_y0 > 0.0)
	chk2 = torch.lt(y[:, :, None], y1[:, None])  # pnts[:, 1][:, None], poly[1:, 1])
	chk3 = torch.sign(diff_)
	# print('chk1 is:\n', chk1)
	# print('chk2 is:\n', chk2)
	# print('chk3 is:\n', chk3)

	pos = (chk1 & chk2 & (chk3 > 0)).sum(axis=2, dtype=int)
	neg = (~chk1 & ~chk2 & (chk3 < 0)).sum(axis=2, dtype=int)
	wn = pos - neg
	# print(wn)
	# print(torch.nonzero(wn))
	# print('with pntss:', pntss)
	out_ = torch.zeros((pntss.shape[0], 3, 2)).to(device)
	idxs = torch.where(wn != 0)
	out_[idxs] = pntss[idxs]

	# out_ = pntss[torch.nonzero(wn)[:, 0]]
	if return_winding:
		return out_, wn
	return out_


def batch_poly_iou(polys1, polys2, sides=4):
	"""
	Return the IoU between two polygons
	:param polys1: [B, N, 2]
	:param polys2: [B, N, 2]
	:param sides: number of sides of the polygon, N
	:return: [B]
	"""
	device = polys1.device
	b = polys1.shape[0]

	polys1 = batch_clockify(polys1)
	polys2 = batch_clockify(polys2)

	xy1 = polys1.unsqueeze(2).expand([-1, -1, polys2.shape[1], -1])
	xy3 = polys2.unsqueeze(1).expand([-1, polys1.shape[1], -1, -1])

	xy2 = polys1.roll(1, 1).unsqueeze(2).expand([-1, -1, polys2.shape[1], -1])
	xy4 = polys2.roll(1, 1).unsqueeze(1).expand([-1, polys1.shape[1], -1, -1])

	x1 = xy1[:, :, :, 0]
	y1 = xy1[:, :, :, 1]
	x2 = xy2[:, :, :, 0]
	y2 = xy2[:, :, :, 1]
	x3 = xy3[:, :, :, 0]
	y3 = xy3[:, :, :, 1]
	x4 = xy4[:, :, :, 0]
	y4 = xy4[:, :, :, 1]

	D = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)

	# resolve D=0 by adding a small delta
	Nx = ((x1 * y2 - x2 * y1) * (x3 - x4) - (x3 * y4 - x4 * y3) * (x1 - x2)) / (D + 1e-10)
	Ny = ((x1 * y2 - x2 * y1) * (y3 - y4) - (x3 * y4 - x4 * y3) * (y1 - y2)) / (D + 1e-10)

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

	a1 = batch_poly_area(polys1)
	a2 = batch_poly_area(polys2)
	# ai = torch.empty(a1.shape).to('cuda')

	polys1_np_keep = batch_torch_wn(polys1, polys2, sides=sides)
	polys2_np_keep = batch_torch_wn(polys2, polys1, sides=sides)

	keep = torch.where(s_total.reshape(b, -1) != 0)

	Nx = Nx.reshape(b, -1)
	Ny = Ny.reshape(b, -1)

	Nxy = torch.cat((Nx[..., None], Ny[..., None]), dim=-1)

	intersections = torch.zeros((b, sides * sides, 2)).to(device)

	intersections[keep] = Nxy[keep]

	union = torch.cat((polys1_np_keep, polys2_np_keep, intersections), dim=1)

	comb = union.abs().mean(dim=-1)

	i = torch.argsort(comb)

	union = union[torch.arange(i.shape[0])[:, None], i]

	new_int = torch.zeros((b, sides * 2, 2)).to(device)

	new_int = union[:, -sides * 2:, :]

	comb = new_int.abs().mean(dim=-1)

	max = torch.max(comb, dim=1, keepdim=True)[1].reshape(-1)

	head = torch.arange(b).to(device)

	cat = torch.cat([head, max], dim=0)
	cat = torch.split(cat, b, dim=0)

	new_int = new_int.double()

	alt = new_int[cat]

	alt = alt.unsqueeze(1).repeat(1, sides * 2, 1).detach()

	idxs = torch.where(comb == 0)

	new_int[idxs] = alt[idxs]

	polyi = batch_clockify(new_int)

	ai = batch_poly_area(polyi)

	iou = ai / (a1 + a2 - ai + 1e-10)

	# outlier = torch.logical_or(torch.logical_or(iou > 1, iou < 0), torch.isnan(iou))
	# if outlier.any():
	#     plt.scatter(polys1[outlier, ::][0, :, 0].cpu().detach().numpy(), polys1[outlier, ::][0, :, 1].cpu().detach().numpy(), color="r")
	#     plt.scatter(polys2[outlier, ::][0, :, 0].cpu().detach().numpy(), polys2[outlier, ::][0, :, 1].cpu().detach().numpy(), color="b")
	#     plt.show()
	#     print(polys1[outlier, ::][0])
	#     print(polys2[outlier, ::][0])
	#     # print(polys1_np_keep[outlier, ::])
	#     # print(polys2_np_keep[outlier, ::])
	#     # print(intersections[outlier, ::])
	#     # print(polys1[outlier, :, :])
	#     # print(polys2[outlier, :, :])
	#     # print(polyi[outlier, :, :])
	#     # print(ai[outlier])
	#     # print(a1[outlier])
	#     # print(a2[outlier])
	#     print(iou[outlier][0])

	return iou


def batch_intersection(polys1, polys2):
	# plt.scatter(polys1[0, :, 0], polys1[0, :, 1], color='b')
	# plt.scatter(polys2[0, :, 0], polys2[0, :, 1], color='r')
	# plt.show()
	device = polys1.device
	b = polys1.shape[0]

	xy1 = polys1.unsqueeze(2).expand([-1, -1, polys2.shape[1], -1])
	xy3 = polys2.unsqueeze(1).expand([-1, polys1.shape[1], -1, -1])

	xy2 = polys1.roll(1, 1).unsqueeze(2).expand([-1, -1, polys2.shape[1], -1])
	xy4 = polys2.roll(1, 1).unsqueeze(1).expand([-1, polys1.shape[1], -1, -1])

	x1 = xy1[:, :, :, 0]
	y1 = xy1[:, :, :, 1]
	x2 = xy2[:, :, :, 0]
	y2 = xy2[:, :, :, 1]
	x3 = xy3[:, :, :, 0]
	y3 = xy3[:, :, :, 1]
	x4 = xy4[:, :, :, 0]
	y4 = xy4[:, :, :, 1]

	D = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)

	# resolve D=0 by adding a small delta
	Nx = ((x1 * y2 - x2 * y1) * (x3 - x4) - (x3 * y4 - x4 * y3) * (x1 - x2)) / (D + 1e-10)
	Ny = ((x1 * y2 - x2 * y1) * (y3 - y4) - (x3 * y4 - x4 * y3) * (y1 - y2)) / (D + 1e-10)

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

	polys1_np_keep = batch_torch_wn_triangle(polys1, polys2)
	polys2_np_keep = batch_torch_wn_triangle(polys2, polys1)

	keep = torch.where(s_total.reshape(b, -1) != 0)

	Nx = Nx.reshape(b, -1)
	Ny = Ny.reshape(b, -1)

	Nxy = torch.cat((Nx[..., None], Ny[..., None]), dim=-1)

	intersections = torch.zeros((b, 9, 2)).to(device)

	intersections[keep] = Nxy[keep]

	union = torch.cat((polys1_np_keep, polys2_np_keep, intersections), dim=1)

	comb = union.abs().mean(dim=-1)

	i = torch.argsort(comb)

	union = union[torch.arange(i.shape[0])[:, None], i]

	new_int = torch.zeros((b, 9, 2)).to(device)

	new_int = union[:, 6:, :]

	comb = new_int.abs().mean(dim=-1)

	max = torch.max(comb, dim=1, keepdim=True)[1].reshape(-1)

	head = torch.arange(b).to(device)

	cat = torch.cat([head, max], dim=0)
	cat = torch.split(cat, b, dim=0)

	new_int = new_int.double()

	alt = new_int[cat]

	alt = alt.unsqueeze(1).repeat(1, 9, 1).detach()

	idxs = torch.where(comb == 0)

	new_int[idxs] = alt[idxs]

	polyi = batch_clockify(new_int)

	ai = batch_poly_area(polyi)

	return ai


def batch_unconvex_poly_iou(polys1, polys2):
	polys1 = batch_clockify(polys1)
	polys2 = batch_clockify(polys2)
	a1 = batch_poly_area(polys1)
	a2 = batch_poly_area(polys2)

	polys1a = polys1.clone()[:, 0:3, :]
	polys1b = polys1.clone()[:, [2, 3, 0], :]
	polys2a = polys2.clone()[:, 0:3, :]
	polys2b = polys2.clone()[:, [2, 3, 0], :]

	ai = batch_intersection(polys1a, polys2a) + batch_intersection(polys1a, polys2b) + \
		batch_intersection(polys1b, polys2a) + batch_intersection(polys1b, polys2b)
	# print(ai)
	# print(a1)
	# print(a2)

	iou = ai / (a1 + a2 - ai + 1e-10)
	return iou


def batch_poly_diou_loss(polys1, polys2, a=1, sides=4):
	"""
	Return the DIoU loss between two polygons
	:param polys1: [B, N, 2]
	:param polys2: [B, N, 2]
	:param a: the weight of the distance part of DIoU loss
	:param sides: number of sides of the polygon, N
	:return: [B]
	"""
	iou = batch_poly_iou(polys1, polys2, sides=sides)
	x_min = torch.min(torch.min(polys1[:, :, 0], dim=1)[0], torch.min(polys2[:, :, 0], dim=1)[0])
	x_max = torch.max(torch.max(polys1[:, :, 0], dim=1)[0], torch.max(polys2[:, :, 0], dim=1)[0])
	y_min = torch.min(torch.min(polys1[:, :, 1], dim=1)[0], torch.min(polys2[:, :, 1], dim=1)[0])
	y_max = torch.max(torch.max(polys1[:, :, 1], dim=1)[0], torch.max(polys2[:, :, 1], dim=1)[0])

	c_sqd = torch.sqrt(torch.square(x_max - x_min) + torch.square(y_max - y_min))

	poly_1_mid = torch.mean(polys1, dim=1)
	poly_2_mid = torch.mean(polys2, dim=1)

	d_sqd = torch.sqrt(torch.sum(torch.square(poly_1_mid - poly_2_mid), dim=1))

	return 1 - iou + a * d_sqd / (c_sqd + 1e-10)


def c_poly_diou_loss(poly1, poly2):
	"""
    Calculate the distance iou loss between two convex polygons
    poly1,poly2 - [n_vertices,2] tensor of x,y,coordinates for each convex polygon
    """
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

	return 1 - iou + d_sqd / (c_sqd + 1e-10)


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
