import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

# 讀取 npy 檔案
data = np.load('./my_file.npz')

pose_list = data['arr1']
finish_pose_list = data['arr2']
# pose_list = np.unique(pose_list.view(np.dtype((np.void, pose_list.dtype.itemsize * pose_list.shape[1])))).view(
#     pose_list.dtype).reshape(-1, pose_list.shape[1])
finish_pose_list = np.unique(
    finish_pose_list.view(np.dtype((np.void, finish_pose_list.dtype.itemsize * finish_pose_list.shape[1])))).view(
    finish_pose_list.dtype).reshape(-1, finish_pose_list.shape[1])

finish_x = finish_pose_list[:, 0]
finish_y = finish_pose_list[:, 1]
print(pose_list)
pose_x = pose_list[:, 0]
pose_y = pose_list[:, 1]
print(pose_x, pose_y)

fig, ax = plt.subplots()
_ = Rectangle((-2.5, 0), 0.2, 0.2, linewidth=1, edgecolor='r', facecolor='none')
ax.add_patch(_)
for i in range(len(finish_x)):
    square = Rectangle((finish_x[i], finish_y[i]), 0.2, 0.2, linewidth=1, edgecolor='r', facecolor='none')
    ax.add_patch(square)

plt.scatter(finish_x, finish_y)
plt.plot(pose_x, pose_y)
plt.show()
