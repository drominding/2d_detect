import numpy as np
 
# 关键点数据
keypoints = [352, 311, 2, 0, 0, 0, 348, 309, 2, 356, 311, 2, 343, 310, 2, 362, 329, 2, 336, 329, 2, 373, 348, 2, 342, 354, 2, 354, 347, 2, 366, 353, 2, 365, 378, 2, 343, 379, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
 

keypoints_reshaped = np.array(keypoints).reshape((-1, 3))

# delet_index = []
# for i in keypoints_reshaped:
#     if i == np.array([0,0,0]):
#         delet_index.append(i)
# print(delet_index)
# 输出结果
# 使用布尔索引删除所有[0,0,0]的行
keypoints_filtered = keypoints_reshaped[~(keypoints_reshaped == [0, 0, 0]).all(axis=1)]
 
# 输出结果
print(keypoints_filtered)
keypoints_xy = keypoints_filtered[:, :2]
 
# 输出结果
print(keypoints_xy)
combined_data = np.concatenate((keypoints_xy, keypoints_xy), axis=0)
print(combined_data)