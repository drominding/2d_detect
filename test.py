import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import cv2  # 用于读取和处理图像
import utils
import point_cloud_utils as pcu

point1 = np.load('data/video2img/0.npy')
point2 = np.load('data/video2img/2.npy')
translated_a = point1.copy()  # 复制原始数组以避免修改它
translated_a[:, 0] += 100
print(translated_a)
hd = pcu.chamfer_distance(point1, point2)
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(translated_a[:,0],translated_a[:,1],translated_a[:,2], c='r', marker='o')
ax.scatter(point1[:,0],point1[:,1],point1[:,2], c='r', marker='o')
print(hd)
plt.show()
