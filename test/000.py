import open3d as o3d
import numpy as np
 
# 生成一些随机点云数据作为示例
points = np.random.rand(1000, 3)  # 创建一个1000x3的数组，包含随机生成的点坐标
 
# 将numpy数组转换为Open3D的点云对象
pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(points)
 
# 可视化点云
o3d.visualization.draw_geometries([pcd])