import torch
import cv2
from fixed_polygon_iou_loss import batch_poly_diou_loss, batch_poly_iou, batch_unconvex_poly_iou
from matplotlib import pyplot as plt
import numpy as np
from tqdm import tqdm

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)


def batch_mid_2_points(mids):
    mid = mids[:, 0:2]
    first_quadrant = mids[:, 2:4]
    second_quadrant = mids[:, 4:6]
    return torch.cat((mid + first_quadrant, mid - second_quadrant, mid - first_quadrant, mid + second_quadrant), dim=0)


def single_test(iterations=100, scale=100):
    target = torch.normal(scale, scale / 4, size=(32, 6))
    pred = torch.normal(scale, scale / 4, size=(32, 6))
    target = batch_mid_2_points(target).to(device)
    pred = batch_mid_2_points(pred).to(device)

    pred_piou = torch.clone(pred).to(device)
    pred_l1 = torch.clone(pred).to(device)
    pred_comb = torch.clone(pred).to(device)

    pred_piou = torch.autograd.Variable(pred_piou, requires_grad=True).to(device)
    pred_l1 = torch.autograd.Variable(pred_l1, requires_grad=True).to(device)
    pred_comb = torch.autograd.Variable(pred_comb, requires_grad=True).to(device)

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

    for k in tqdm(range(iterations)):
        opt_piou.zero_grad()
        loss = batch_poly_diou_loss(pred_piou.view(-1, 4, 2), target.view(-1, 4, 2)).mean()

        iou = batch_poly_iou(pred_piou.view(-1, 4, 2), target.view(-1, 4, 2)).cpu().detach().numpy().mean()
        loss_history_piou.append(iou)
        loss.backward()

        opt_piou.step()
        scheduler_piou.step(loss)
        del loss

        opt_l1.zero_grad()
        l1_loss = torch.abs(pred_l1 - target).mean() / scale
        loss_history_l1.append(batch_poly_iou(pred_l1.view(-1, 4, 2), target.view(-1, 4, 2)).cpu().detach().numpy().mean())
        l1_loss.backward()
        opt_l1.step()
        scheduler_l1.step(l1_loss)
        del l1_loss

        opt_comb.zero_grad()
        c_iou = batch_poly_iou(pred_comb.view(-1, 4, 2), target.view(-1, 4, 2)).cpu().detach()
        c_iou_loss = batch_poly_diou_loss(pred_comb.view(-1, 4, 2), target.view(-1, 4, 2))
        c_l1_loss = torch.abs(pred_comb.view(-1, 8) - target.view(-1, 8)).mean(dim=1)
        comb_loss = k / iterations * c_l1_loss + 0.1 * (1 - k / iterations) * c_iou_loss
        comb_loss = comb_loss.mean()
        loss_history_comb.append(c_iou.numpy().mean())
        comb_loss.backward()
        opt_comb.step()
        scheduler_comb.step(comb_loss)
        del comb_loss
    return loss_history_piou, loss_history_l1, loss_history_comb


loss_history_pious = np.zeros(3000)
loss_history_l1s = np.zeros(3000)
loss_history_combs = np.zeros(3000)
trials = 5
for i in range(trials):
    loss_history_piou, loss_history_l1, loss_history_comb = single_test(3000, 40)
    loss_history_piou = np.array(loss_history_piou)
    loss_history_l1 = np.array(loss_history_l1)
    loss_history_comb = np.array(loss_history_comb)
    loss_history_pious = loss_history_pious + loss_history_piou
    loss_history_l1s = loss_history_l1s + loss_history_l1
    loss_history_combs = loss_history_combs + loss_history_comb
    print(loss_history_pious[-1], loss_history_l1s[-1], loss_history_combs[-1])
loss_history_pious /= trials
loss_history_l1s /= trials
loss_history_combs /= trials
# np.savetxt('results/convergence/four/pious.txt', loss_history_pious)
# np.savetxt('results/convergence/four/l1s.txt', loss_history_l1s)
# np.savetxt('results/convergence/four/combs.txt', loss_history_combs)

plt.plot(loss_history_pious, c='b')
plt.plot(loss_history_l1s, c='r')
plt.plot(loss_history_combs, c='g')
plt.legend(['piou', 'l1', 'piou+l1'])
plt.ylabel('iou')
plt.xlabel('iterations')
plt.show()
