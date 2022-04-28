from data.Kitti import Kitti

dataset = Kitti(root="..")
# dataset.__getitem__(25)
dataset.plot(25)
# dataset.plot(2001)

