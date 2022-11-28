import torch
import cv2
from fixed_polygon_iou_loss import batch_poly_diou_loss, batch_poly_iou
from matplotlib import pyplot as plt
import numpy as np


def batch_mid_2_points(mids):
	mid = mids[:, 0:2]
	first_quadrant = mids[:, 2:4]
	second_quadrant = mids[:, 4:6]
	return torch.cat((mid + first_quadrant, mid - second_quadrant, mid - first_quadrant, mid + second_quadrant), dim=0)


def single_test(iterations=1000, scale=100):
	target = torch.normal(scale, scale / 4, size=(32, 6))
	pred = torch.normal(scale, scale / 4, size=(32, 6))
	target = batch_mid_2_points(target)
	pred = batch_mid_2_points(pred)

	pred_piou = torch.clone(pred)
	pred_l1 = torch.clone(pred)
	pred_comb = torch.clone(pred)

	pred_piou = torch.autograd.Variable(pred_piou, requires_grad=True)
	pred_l1 = torch.autograd.Variable(pred_l1, requires_grad=True)
	pred_comb = torch.autograd.Variable(pred_comb, requires_grad=True)

	opt_piou = torch.optim.Adam([pred_piou], lr=0.02)
	opt_l1 = torch.optim.Adam([pred_l1], lr=0.02)
	opt_comb = torch.optim.Adam([pred_comb], lr=0.02)

	# scheduler_piou = torch.optim.lr_scheduler.ReduceLROnPlateau(opt_piou, factor=0.2, patience=200, threshold=0.001, min_lr=0.0001)
	# scheduler_l1 = torch.optim.lr_scheduler.ReduceLROnPlateau(opt_l1, factor=0.2, patience=200, threshold=0.001, min_lr=0.0001)
	# scheduler_comb = torch.optim.lr_scheduler.ReduceLROnPlateau(opt_comb, factor=0.2, patience=200, threshold=0.001, min_lr=0.0001)
	scheduler_piou = torch.optim.lr_scheduler.MultiStepLR(opt_piou, milestones=[1500, 2000, 2500], gamma=0.1)
	scheduler_l1 = torch.optim.lr_scheduler.MultiStepLR(opt_l1, milestones=[1500, 2000, 2500], gamma=0.1)
	scheduler_comb = torch.optim.lr_scheduler.MultiStepLR(opt_comb, milestones=[1500, 2000, 2500], gamma=0.1)

	loss_history_piou = []
	loss_history_l1 = []
	loss_history_comb = []

	for k in range(iterations):
		opt_piou.zero_grad()
		loss = batch_poly_diou_loss(pred_piou.view(-1, 4, 2), target.view(-1, 4, 2)).mean()

		iou = batch_poly_iou(pred_piou.view(-1, 4, 2), target.view(-1, 4, 2)).detach().numpy().mean()
		loss_history_piou.append(iou)
		loss.backward()

		opt_piou.step()
		scheduler_piou.step(loss)
		del loss

		opt_l1.zero_grad()
		l1_loss = torch.abs(pred_l1 - target).mean()
		loss_history_l1.append(batch_poly_iou(pred_l1.view(-1, 4, 2), target.view(-1, 4, 2)).detach().numpy().mean())
		l1_loss.backward()
		opt_l1.step()
		scheduler_l1.step(l1_loss)
		del l1_loss

		opt_comb.zero_grad()
		comb_loss = torch.abs(pred_comb - target).mean() / scale + batch_poly_diou_loss(pred_comb.view(-1, 4, 2), target.view(-1, 4, 2)).mean()
		loss_history_comb.append(batch_poly_iou(pred_comb.view(-1, 4, 2), target.view(-1, 4, 2)).detach().numpy().mean())
		comb_loss.backward()
		opt_comb.step()
		scheduler_comb.step(comb_loss)
		del comb_loss
	return loss_history_piou, loss_history_l1, loss_history_comb


loss_history_piou, loss_history_l1, loss_history_comb = single_test(2000, 40)
loss_history_piou = np.array(loss_history_piou)
loss_history_l1 = np.array(loss_history_l1)
loss_history_comb = np.array(loss_history_comb)

plt.plot(loss_history_piou, c='b')
plt.plot(loss_history_l1, c='r')
plt.plot(loss_history_comb, c='g')
plt.legend(['piou', 'l1', 'piou+l1'])
plt.ylabel('iou')
plt.xlabel('iterations')
plt.show()
