import numpy as np

def interpolate_points(point1, point2, step_size=None):
    if step_size == None:
        step_size = 1
    
    # 计算两点之间的总距离
    point1 = np.array(point1)
    point2 = np.array(point2)
    total_distance = np.linalg.norm(point2 - point1)
    
    # 计算需要生成的间隔数量（不包括点a和点b之间的间隔，但我们会加1来包含点a和生成的间隔点）
    # 然后，由于我们需要一个额外的点来包含点b（如果它不是通过间隔直接得到的），
    # 我们实际上要生成num_intervals + 1个点，但num_intervals是基于总距离和间距n来计算的
    num_intervals = int(total_distance // step_size)
    
    # 生成沿着ab连线的等距点
    points = [point1 + i * step_size / total_distance * (point2 - point1) for i in range(num_intervals + 1)]
    
    # 注意：由于浮点运算的精度问题，最后一个点可能不是精确的b，但通常这种误差非常小
    # 如果你需要确保最后一个点精确是b，你可以替换上面的列表推导式为以下代码：
    # points = [a + i * n / total_distance * (b - a) for i in range(num_intervals)]
    # points.append(b)  # 直接添加点b作为最后一个点
    
    # 将点转换为NumPy数组（可选）
    points = np.array(points)
    
    return points

def handel_duplication(array):
    points = array
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
    return unique_points



# 示例使用
if __name__ == '__main__':
    point1 = (0, 0, 0)
    point2 = (4, 5, 6)


    points = interpolate_points(point1, point2)
    for p in points:
        print(p)