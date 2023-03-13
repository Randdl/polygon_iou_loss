import torch
import cv2
from fixed_polygon_iou_loss import batch_poly_diou_loss, batch_poly_iou, batch_unconvex_poly_iou, c_poly_diou_loss, \
    c_poly_iou
from matplotlib import pyplot as plt
import numpy as np
from tqdm import tqdm


def batch_mid_2_points_eight(mids):
    mid = mids[:, 0:2]
    first_quadrant = mids[:, 2:4]
    second_quadrant = mids[:, 4:6]
    third_quadrant = mids[:, 6:8]
    fourth_quadrant = mids[:, 8:10]
    return torch.cat((mid + first_quadrant, mid - second_quadrant, mid - first_quadrant, mid + second_quadrant,
                      mid + third_quadrant, mid - fourth_quadrant, mid - third_quadrant, mid + fourth_quadrant), dim=0)


def single_test_eight(iterations=100, scale=100, size=8):
    target = torch.normal(scale, scale / 4, size=(size, 10))
    pred = torch.normal(scale, scale / 4, size=(size, 10))
    target = batch_mid_2_points_eight(target)
    pred = batch_mid_2_points_eight(pred)

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
    scheduler_piou = torch.optim.lr_scheduler.MultiStepLR(opt_piou, milestones=[1000, 1500, 2000, 2500], gamma=0.1)
    scheduler_l1 = torch.optim.lr_scheduler.MultiStepLR(opt_l1, milestones=[1000, 1500, 2000, 2500], gamma=0.1)
    scheduler_comb = torch.optim.lr_scheduler.MultiStepLR(opt_comb, milestones=[1000, 1500, 2000, 2500], gamma=0.1)

    loss_history_piou = []
    loss_history_l1 = []
    loss_history_comb = []

    for k in tqdm(range(iterations)):
        opt_piou.zero_grad()
        loss = batch_poly_diou_loss(pred_piou.view(-1, 8, 2), target.view(-1, 8, 2), a=0, sides=8).mean()
        # print(loss.mean())
        # loss = torch.zeros(1, requires_grad=False)
        # for i in range(size):
        #     loss += c_poly_diou_loss(pred_piou.view(-1, 8, 2)[i], target.view(-1, 8, 2)[i])
        # loss /= size
        # print(loss)

        iou = batch_poly_iou(pred_piou.view(-1, 8, 2), target.view(-1, 8, 2), sides=8).mean()
        # for i in range(size):
        #     iou += c_poly_iou(pred_piou.view(-1, 8, 2)[i], target.view(-1, 8, 2)[i])
        # iou /= size
        loss_history_piou.append(iou.detach().numpy())
        loss.backward()

        opt_piou.step()
        scheduler_piou.step(loss)
        del loss

        opt_l1.zero_grad()
        l1_loss = torch.abs(pred_l1 - target).mean() / scale
        iou_l1 = batch_poly_iou(pred_l1.view(-1, 8, 2), target.view(-1, 8, 2), sides=8).mean()
        # for i in range(size):
        #     iou_l1 += c_poly_iou(pred_l1.view(-1, 8, 2)[i], target.view(-1, 8, 2)[i])
        # iou_l1 /= size
        # if iou_l1 > 1:
        #     iou_l1 = loss_history_l1[-1]
        #     loss_history_l1.append(iou_l1)
        # else:
        #     loss_history_l1.append(iou_l1.detach().numpy())
        loss_history_l1.append(iou_l1.detach().numpy())
        l1_loss.backward()
        opt_l1.step()
        scheduler_l1.step(l1_loss)
        del l1_loss

        opt_comb.zero_grad()
        c_iou = batch_poly_iou(pred_comb.view(-1, 8, 2), target.view(-1, 8, 2), sides=8).mean()
        # for i in range(size):
        #     c_iou += c_poly_iou(pred_comb.view(-1, 8, 2)[i], target.view(-1, 8, 2)[i])
        # c_iou /= size
        # print(c_iou)
        c_iou_loss = batch_poly_diou_loss(pred_comb.view(-1, 8, 2), target.view(-1, 8, 2), a=0, sides=8).mean()
        # for i in range(size):
        #     c_iou_loss += c_poly_diou_loss(pred_comb.view(-1, 8, 2)[i], target.view(-1, 8, 2)[i])
        # c_iou_loss /= size
        c_l1_loss = torch.abs(pred_comb.view(-1, 8) - target.view(-1, 8)).mean(dim=1)
        comb_loss = k / iterations * c_l1_loss + 0.1 * (1 - k / iterations) * c_iou_loss
        comb_loss = comb_loss.mean() * 2
        loss_history_comb.append(c_iou.detach().numpy())
        comb_loss.backward()
        opt_comb.step()
        scheduler_comb.step(comb_loss)
        del comb_loss
    return loss_history_piou, loss_history_l1, loss_history_comb


loss_history_pious = np.zeros(3000)
loss_history_l1s = np.zeros(3000)
loss_history_combs = np.zeros(3000)
trials = 2
for i in range(trials):
    loss_history_piou, loss_history_l1, loss_history_comb = single_test_eight(3000, 40, 32)
    loss_history_piou = np.array(loss_history_piou)
    loss_history_l1 = np.array(loss_history_l1)
    loss_history_comb = np.array(loss_history_comb)
    print(loss_history_piou[-1], loss_history_l1[-1], loss_history_comb[-1])
    loss_history_pious = loss_history_pious + loss_history_piou
    loss_history_l1s = loss_history_l1s + loss_history_l1
    loss_history_combs = loss_history_combs + loss_history_comb
loss_history_pious /= trials
loss_history_l1s /= trials
loss_history_combs /= trials
# np.savetxt('results/convergence/eight/pious.txt', loss_history_pious)
# np.savetxt('results/convergence/eight/l1s.txt', loss_history_l1s)
# np.savetxt('results/convergence/eight/combs.txt', loss_history_combs)

plt.plot(loss_history_pious, c='b')
plt.plot(loss_history_l1s, c='r')
plt.plot(loss_history_combs, c='g')
plt.legend(['piou', 'l1', 'piou+l1'])
plt.ylabel('iou')
plt.xlabel('iterations')

plt.rcParams["font.size"] = "70"
plt.savefig('Basic.png', dpi=300, bbox_inches="tight")
plt.show()
