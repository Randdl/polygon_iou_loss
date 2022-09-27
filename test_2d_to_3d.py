import numpy as np
import torch
from matplotlib import pyplot as plt
import plotly.graph_objects as go
from scipy.optimize import least_squares

from data.Kitti import Kitti, computeBox3D, load_dataset_detectron2, batch_computeBox3D, np_computeBox3D

dataset = Kitti(root="..")
# dataset.__getitem__(25)
# dataset.plot(5)
image = dataset.__getitem__(1)
target = image['target'][0]
# print(target['base'])
# print(target['corners'])
origin3d = target['origin3d']
print(target['type'])
origin3d = np.array(origin3d)
origin3d_tensor = torch.tensor(origin3d, dtype=torch.float)
print(origin3d)
base_3Dto2D, corners_2D, corners_3D, bb2d_lines_verts, depth = computeBox3D([x+0.01 for x in origin3d], target['calib'])
# print(base_3Dto2D)
# load_dataset_detectron2(test=False)
w_range = np.arange(0.76, 4.2, 0.5)
h_range = np.arange(0.3, 3, 0.5)
l_range = np.arange(0.2, 35, 1)
x_range = np.arange(-44, 40, 1)
y_range = np.arange(-2, 6, 1)
z_range = np.arange(-4, 147, 1)
ry_range = np.arange(-3.14, 3.14, 0.5)

small_batch = torch.reshape(torch.tensor([[3, 2, 4, 10, 3, 10, 1], [3, 2, 20, 10, 3, 10, 1]], dtype=torch.float), (2, 7))
corners_2D, base_3Dto2D = batch_computeBox3D(origin3d_tensor.repeat(2, 1), torch.tensor(target['calib'], dtype=torch.float))
plt.scatter(x=corners_2D[0, 0, :], y=corners_2D[0, 1, :], s=40, color="r")
plt.show()

P = target['calib']
real_corners, _ = np_computeBox3D(origin3d, P)


def diff_fun(input):
    corners, _ = np_computeBox3D(input, P)
    diff = real_corners - corners
    return diff.flatten()


x0 = np.array([3, 2, 20, 30, 5, 80, 2])
# x0 = origin3d + (np.random.rand(7) - 0.5)
bounds = np.array([0.76, 0.3, 0.2, -44, -2, -4, -3.14]), np.array([4.2, 3, 35, 40, 6, 147, 3.14])
res_1 = least_squares(diff_fun, x0, bounds=bounds)
print(res_1.x)

real_corners, c1 = np_computeBox3D(origin3d, P)
optimal_corners, c2 = np_computeBox3D(res_1.x, P)
print(real_corners)
print(optimal_corners)
fig = plt.figure()
ax = plt.axes(projection='3d')
ax.scatter3D(c1[0], c1[1], c1[2], cmap='Greens')
ax.scatter3D(c2[0], c2[1], c2[2], cmap='Greens')
plt.show()
z = [0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1]
fig = go.Figure(data=[go.Scatter3d(
    x=np.concatenate([c1[0], c2[0]]),
    y=np.concatenate([c1[1], c2[1]]),
    z=np.concatenate([c1[2], c2[2]]),
    mode='markers',
    marker=dict(
        size=12,
        color=z,                # set color to an array/list of desired values
        colorscale='Viridis',   # choose a colorscale
        opacity=0.8
    )
)])
fig.update_layout(margin=dict(l=0, r=0, b=0, t=0))
fig.show()
