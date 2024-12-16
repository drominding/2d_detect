


# 假设给定的点列表为points，每个点是一个[x, y, z]的列表
points = [
        [1,1],[1,1],[1,1],
        [1,2],[1,2],[1,2],
        [1,3],[1,3],[1,1],
        [1,4]
]


# 下面是一个简化的版本，只记录z值：
 
# 重置point_dict和processed_points
point_dict = {}
processed_points = []
 
# 再次遍历每个点
for point in points:
    x, y = point
    # 检查点是否已出现过
    if (x, y) in point_dict:
        # 如果已出现，则获取当前的z值，并增加0.1
        z = point_dict[(x, y)] + 0.0001
        # 更新字典中的z值
        point_dict[(x, y)] = z
    else:
        # 如果未出现，则初始化z值为0
        z = 0
        # 将z值添加到字典中
        point_dict[(x, y)] = z
    # 将处理后的三维点添加到列表中
    processed_points.append((x, y, z))
 
# 输出处理后的点列表

print(processed_points)