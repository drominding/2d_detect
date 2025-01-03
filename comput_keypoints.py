'''
比较两张图片关键点连线的角度
'''

import cv2
import numpy as np
import mediapipe as mp
import torch

# 初始化MediaPipe的姿态模型
mp_pose = mp.solutions.pose
pose = mp_pose.Pose()

# 加载两张图片
image1 = cv2.imread(r'E:\CODE\2D_detect\data\stu_1.png')
image2 = cv2.imread(r'E:\CODE\2D_detect\data\side.jpg')

# 将图片转换为RGB格式，因为MediaPipe需要RGB格式的图片
image1_rgb = cv2.cvtColor(image1, cv2.COLOR_BGR2RGB)
image2_rgb = cv2.cvtColor(image2, cv2.COLOR_BGR2RGB)

# 对第一张图片进行关键点检测
results1 = pose.process(image1_rgb)

# 对第二张图片进行关键点检测
results2 = pose.process(image2_rgb)

# 将关键点绘制在原始图片上（注意：这里需要转换回BGR格式以显示）
image1_with_keypoints = cv2.cvtColor(image1_rgb, cv2.COLOR_RGB2BGR)
image2_with_keypoints = cv2.cvtColor(image2_rgb, cv2.COLOR_RGB2BGR)

# 绘制第一张图片的关键点
if results1.pose_landmarks:
    mp.solutions.drawing_utils.draw_landmarks(
        image1_with_keypoints, results1.pose_landmarks, mp_pose.POSE_CONNECTIONS)

# 绘制第二张图片的关键点
if results2.pose_landmarks:
    mp.solutions.drawing_utils.draw_landmarks(
        image2_with_keypoints, results2.pose_landmarks, mp_pose.POSE_CONNECTIONS)
    


'''
计算动作相似程度
    对于结果进行加工，仅保留手臂和双腿的关键点
    左边的手: 12-14, 12,16
    右边的手: 11-13, 11-15
    左边的脚: 24-16, 24-28
    右边的脚: 23-25, 25-27
    左边的躯干: 12-24
    右边的躯干: 11,23
'''
def cosine_similarity(vec1, vec2):
    # 将向量转换为NumPy数组（如果它们还不是的话）
    vec1 = np.array(vec1)
    vec2 = np.array(vec2)
    
    # 归一化向量
    vec1_norm = vec1 / np.linalg.norm(vec1)
    vec2_norm = vec2 / np.linalg.norm(vec2)
    
    # 计算点积作为余弦相似度
    cos_sim = np.dot(vec1_norm, vec2_norm)
    
    return cos_sim


def calculating_similarity(fix_point,free_point,student_results,teacher_results,student_img,teacher_img):
    '''
    计算两组节点的相似程度
    num1是定点,num2是动点,result1是学生结果,result2是参考结果
    '''
    student_results = student_results.pose_landmarks.landmark
    teacher_results = teacher_results.pose_landmarks.landmark
    student_x = student_results[free_point].x - student_results[fix_point].x
    student_y = student_results[free_point].y - student_results[fix_point].y
    student_z = student_results[free_point].z - student_results[fix_point].z
    sh, sw, sc = student_img.shape
    teacher_x = teacher_results[free_point].x - teacher_results[fix_point].x
    teacher_y = teacher_results[free_point].y - teacher_results[fix_point].y
    teacher_z = teacher_results[free_point].z - teacher_results[fix_point].z
    th, tw, tc = teacher_img.shape
    vec1 = [student_x*sw,student_y*sh,student_z*sw/5]
    vec2 = [teacher_x*tw,teacher_y*th,teacher_z*tw/5]
    # vec1 = [student_x,student_y,student_z]
    # vec2 = [teacher_x,teacher_y,teacher_z]
    return cosine_similarity(vec1,vec2)


anser = calculating_similarity(12,14,results1,results2,image1,image2)
print(anser)





# 显示结果图片
cv2.imshow('Image 1 with Keypoints', image1_with_keypoints)
cv2.imshow('Image 2 with Keypoints', image2_with_keypoints)

# 等待按键输入以关闭窗口
cv2.waitKey(0)
cv2.destroyAllWindows()