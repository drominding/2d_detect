# 假设给定的点列表为points，每个点是一个[x, y, z]的列表
points = [
        [1,1,0],[1,1,0],[1,1,0],[1,1,0],
        [1,2,0],[1,2,0],[1,2,0],
        [1,3,0],[1,3,0],
        [1,4,0]
]

# 由于z值都相等，我们可以只考虑(x, y)坐标
xy_coords = [(x, y) for x, y, z in points]

# 使用一个字典来记录每个(x, y)坐标出现的次数
xy_count = {}
for xy in xy_coords:
    if xy in xy_count:
        xy_count[xy] += 1
    else:
        xy_count[xy] = 1

# 处理重复的点
unique_points = []
for x, y, z in points:
    xy = (x, y)
    if xy_count[xy] > 1:  # 如果(x, y)是重复的
        # 对z值进行微小的偏移（这里假设z的原始值对于所有点是z0，我们逐个增加0.1）
        # 注意：这里的z_offset应该是一个递增的变量，而不是固定的0.1，
        # 以确保每个重复的点都有一个唯一的z值。但由于题目要求“逐个+0.1”，
        # 我们这里简化处理，只展示概念。在实际应用中，可能需要更复杂的逻辑来分配z值。
        z_offset = 0.00001 * (xy_count[xy] - 1)  # 计算偏移量（这里只是示例，可能不是实际想要的逻辑）
        # 但由于我们假设所有原始z值相同，这里我们直接加0.1（或递增的偏移量）来区分
        new_z = z + z_offset  # 注意：这里可能会超出原始z值的范围，需要根据实际情况调整
        unique_points.append([x, y, new_z])
        xy_count[xy] -= 1  # 减少该(x, y)坐标的计数（为了模拟“逐个”处理的效果）
    else:
        unique_points.append([x, y, z])  # (x, y)不重复，直接添加到结果列表中



# 输出处理后的点列表
for point in unique_points:
    print(point)