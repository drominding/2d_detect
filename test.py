import cv2
import mediapipe as mp

# 初始化MediaPipe手部模型
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils

# 尝试打开索引为1的摄像头（根据你的系统配置可能有所不同）
cap = cv2.VideoCapture("h1.jpg")


while cap.isOpened():
    success, image = cap.read()
    if not success:
        print("忽略空帧")
        continue

    # 处理图像并检测手部关键点
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = hands.process(image_rgb)

    # 绘制手部关键点
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(image, hand_landmarks, mp_hands.HAND_CONNECTIONS)

    # 显示图像
    cv2.imshow('Hand Tracking', image)

    # 按下'q'键退出循环
    if cv2.waitKey(5) & 0xFF == ord('q'):
        break

# 释放资源
cap.release()
cv2.destroyAllWindows()