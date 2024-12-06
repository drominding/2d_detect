import cv2
import mediapipe as mp
import numpy as np

# 初始化MediaPipe的姿态模型
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.5)

# 加载两张图片
image1 = cv2.imread('h1.jpg')
image2 = cv2.imread('h4.jpg')

# 将图片转换为RGB格式，因为MediaPipe需要RGB格式的图片
image1_rgb = cv2.cvtColor(image1, cv2.COLOR_BGR2RGB)
image2_rgb = cv2.cvtColor(image2, cv2.COLOR_BGR2RGB)

# 对第一张图片进行关键点检测
results1 = hands.process(image1_rgb)

# 对第二张图片进行关键点检测
results2 = hands.process(image2_rgb)

'''
关键点检测开始
'''




# 提取关键点坐标（如果检测到关键点）
keypoints1 = []
if results1.multi_hand_landmarks:
    for hand_landmarks  in results1.multi_hand_landmarks:
        for landmark in hand_landmarks.landmark:
            keypoints1.append((landmark.x, landmark.y))
print(keypoints1) 
keypoints2 = []
if results2.multi_hand_landmarks:
    for hand_landmarks  in results2.multi_hand_landmarks:
        for landmark in hand_landmarks.landmark:
            keypoints2.append((landmark.x, landmark.y))
print(keypoints2)  
# 确保两张图片都检测到了相同数量的关键点
if len(keypoints1) != len(keypoints2):
    print("The number of keypoints detected in both images is not the same.")
    exit()
 
# 计算关键点之间的欧氏距离，并求和作为相似度度量（距离越小，相似度越高）
similarity_score = sum(np.linalg.norm(np.array(kp1) - np.array(kp2)) for kp1, kp2 in zip(keypoints1, keypoints2))
 
# 输出相似度评分（注意：这个评分是距离的和，所以是一个较大的数；你可以根据需要对其进行归一化或反转）
print(f"Similarity score (lower is more similar): {similarity_score}")

'''
关键点检测结束
'''

# 将关键点绘制在原始图片上（注意：这里需要转换回BGR格式以显示）
image1_with_keypoints = cv2.cvtColor(image1_rgb, cv2.COLOR_RGB2BGR)
image2_with_keypoints = cv2.cvtColor(image2_rgb, cv2.COLOR_RGB2BGR)
mp_drawing1 = mp.solutions.drawing_utils
# 绘制第一张图片的关键点
if results1.multi_hand_landmarks:
    for hand_landmarks in results1.multi_hand_landmarks:
        mp_drawing1.draw_landmarks(image1_with_keypoints, hand_landmarks, mp_hands.HAND_CONNECTIONS)
# 绘制第二张图片的关键点
if results2.multi_hand_landmarks:
    for hand_landmarks in results2.multi_hand_landmarks:
        mp.solutions.drawing_utils.draw_landmarks(
            image2_with_keypoints, hand_landmarks, mp_hands.HAND_CONNECTIONS)

# 显示结果图片
cv2.imshow('Image 1 with Keypoints', image1_with_keypoints)
cv2.imshow('Image 2 with Keypoints', image2_with_keypoints)

# 等待按键输入以关闭窗口
cv2.waitKey(0)
cv2.destroyAllWindows()