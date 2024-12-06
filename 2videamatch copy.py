# 线程不安全，错误示范
import cv2
import mediapipe as mp
import threading

# 初始化MediaPipe Pose模型
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=False, model_complexity=1, enable_segmentation=False, min_detection_confidence=0.5)

# 绘制关键点函数
def draw_pose(image, results):
    if not results.pose_landmarks:
        return image
    mp.solutions.drawing_utils.draw_landmarks(
        image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
    return image

# 处理视频流的函数
def process_video(video_path, window_name):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video {video_path}")
        return

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # 将图像从BGR转换为RGB
        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        # 处理图像并获取姿势结果
        results = pose.process(image_rgb)
        # 绘制姿势关键点
        image_with_pose = draw_pose(frame, results)

        # 显示结果图像
        cv2.imshow(window_name, image_with_pose)

        # 按'q'键退出
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyWindow(window_name)

# 主函数
def main():
    video_path1 = 'student_video_30fps.mp4'
    video_path2 = 'teacher_video_30fps.mp4'

    # 创建线程处理两个视频流
    thread1 = threading.Thread(target=process_video, args=(video_path1, 'Video 1 Pose'))
    thread2 = threading.Thread(target=process_video, args=(video_path2, 'Video 2 Pose'))

    # 启动线程
    thread1.start()
    thread2.start()

    # 等待线程完成
    thread1.join()
    thread2.join()

if __name__ == "__main__":
    main()