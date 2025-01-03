import cv2
import numpy as np

# 读取深度图像
image_path = '1698815928.213450.png'  # 替换为你的深度图像文件路径
image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
print(image.shape)
print(image)
# 计算缩放因子
scale_factor = 255.0 / 2635.0
 
# 应用缩放因子
scaled_data = (image * scale_factor).astype(np.uint8)

# 检查图像是否成功加载
if image is None:
    print(f"Error: Unable to load depth image at {image_path}")
else:
    # 显示深度图像
    cv2.imshow('Depth Image', scaled_data)

    # 等待按键事件，按任意键关闭窗口
    cv2.waitKey(0)
    cv2.destroyAllWindows()