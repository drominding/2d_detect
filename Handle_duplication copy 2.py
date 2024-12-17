import numpy as np

# 假设给定的点列表已经是 NumPy 数组
points = np.array([(1, 2, 3), (1, 2, 3), (1, 3, 3), (1, 4, 3)])

# 创建一个字典来存储每个 (x, y) 坐标的当前 z 偏移量
offset_dict = {}

# 创建一个列表来存储处理后的点
unique_points = []

# 遍历原始点数组
for point in points:
    x, y, z = point
    xy = (x, y)
    
    # 检查 (x, y) 坐标是否已经在字典中
    if xy in offset_dict:
        # 如果是重复的，使用当前的偏移量来增加 z 值
        offset = offset_dict[xy]
        new_z = z + offset * 0.1
        # 更新偏移量（为下一个重复的点准备）
        offset_dict[xy] += 1
    else:
        # 如果不是重复的，初始化偏移量为 0，并使用原始的 z 值
        new_z = z
        offset_dict[xy] = 1  # 设置为 1，以便下一个重复的点可以增加偏移量
    
    # 将处理后的点添加到结果列表中
    unique_points.append([x, y, new_z])

# 将结果转换为 NumPy 数组（如果需要）
unique_points = np.array(unique_points)

# 输出处理后的点列表
for point in unique_points:
    print(point)