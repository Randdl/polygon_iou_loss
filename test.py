import collections
import numpy as np

import torch
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import torch.nn as nn

import data
from data.Kitti import Kitti
import data.func
from data.func import Cropper, Resizer, Normalizer, collater

from retinanet import model

import matplotlib.pyplot as plt

import timeit

from skimage import io, transform

print('CUDA available: {}'.format(torch.cuda.is_available()))


def main(args=None):
    p = transforms.Compose([Cropper(350, 1200), Normalizer()])
    dataset_train = Kitti(root="..", transforms=p)
    # dataloader_train = DataLoader(dataset_train, batch_size=8, num_workers=3, collate_fn=collater, shuffle=True)
    dataloader_train = torch.utils.data.DataLoader(
        dataset_train,
        batch_size=16,
        num_workers=3,
        shuffle=True,
        collate_fn=collater
    )
    retinanet = model.resnet18(num_classes=dataset_train.num_classes(), pretrained=True)
    use_gpu = True
    epochs = 2
    if torch.cuda.is_available():
        retinanet = retinanet.cuda()

    if torch.cuda.is_available():
        retinanet = torch.nn.DataParallel(retinanet).cuda()
    else:
        retinanet = torch.nn.DataParallel(retinanet)

    retinanet.training = True

    optimizer = optim.Adam(retinanet.parameters(), lr=1e-4)

    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=3, verbose=True)

    loss_hist = collections.deque(maxlen=2500)

    retinanet.train()
    retinanet.module.freeze_bn()

    print('Num training images: {}'.format(len(dataset_train)))

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    for epoch_num in range(epochs):

        retinanet.train()
        retinanet.module.freeze_bn()

        epoch_loss = []

        for iter_num, data in enumerate(dataloader_train):
            start.record()

            print("......")
            try:
                optimizer.zero_grad()

                if torch.cuda.is_available():
                    classification_loss, regression_loss = retinanet([data['image'].cuda().float(), data['labels']])
                else:
                    classification_loss, regression_loss = retinanet([data['image'].float(), data['labels']])

                classification_loss = classification_loss.mean()
                regression_loss = regression_loss.mean()

                loss = classification_loss + regression_loss

                if bool(loss == 0):
                    continue

                loss.backward()

                torch.nn.utils.clip_grad_norm_(retinanet.parameters(), 0.1)

                optimizer.step()

                loss_hist.append(float(loss))

                epoch_loss.append(float(loss))

                end.record()
                torch.cuda.synchronize()
                print(
                    'Epoch: {} | Iteration: {} | Classification loss: {:1.5f} | Regression loss: {:1.5f} | Running '
                    'loss: {:1.5f} | Time: {}'.format(
                        epoch_num, iter_num, float(classification_loss), float(regression_loss), np.mean(loss_hist),
                        start.elapsed_time(end)))

                del classification_loss
                del regression_loss
            except Exception as e:
                print(e)
                continue

        scheduler.step(np.mean(epoch_loss))

        torch.save(retinanet.module, '{}_retinanet_{}.pt'.format("Kitti", epoch_num))

    # retinanet.eval()

    torch.save(retinanet, 'model_final_with_{}_epochs_{}Loss.pt'.format(epochs, "polyIOU"))

    plt.title('model_with_{}_epochs_{}Loss'.format(epochs, "polyIOU"))
    plt.plot(loss_hist)
    plt.show()
    plt.savefig('model_with_{}_epochs_{}Loss.png'.format(epochs, "polyIOU"))


if __name__ == '__main__':
    main()
