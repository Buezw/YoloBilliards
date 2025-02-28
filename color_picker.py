import cv2
import numpy as np
import mss
import time

# 定义屏幕捕获区域（根据实际分辨率调整）
monitor = {"top": 240, "left": 640, "width": 1917 - 640, "height": 1160 - 240}

# 全局变量，保存当前捕获的图像和缩放倍数
current_frame = None
scale_factor = 1.5  # 放大倍数

def pick_color(event, x, y, flags, param):
    """
    鼠标点击回调函数：在点击时获取当前图像中该点的颜色
    注意：由于预览图像经过了放大，需要将点击坐标换算回原图坐标
    """
    global current_frame, scale_factor
    if event == cv2.EVENT_LBUTTONDOWN and current_frame is not None:
        # 将点击坐标转换为原始图像的坐标
        orig_x = int(x / scale_factor)
        orig_y = int(y / scale_factor)
        h, w = current_frame.shape[:2]
        if orig_x >= w or orig_y >= h:
            print("点击位置超出图像范围")
            return
        # 获取 BGR 颜色（当前图像为 BGR 格式）
        bgr_color = current_frame[orig_y, orig_x]
        # 将 BGR 转换为 HSV
        hsv_color = cv2.cvtColor(np.uint8([[bgr_color]]), cv2.COLOR_BGR2HSV)[0][0]
        print(f"点击坐标（原图）: ({orig_x}, {orig_y})")
        print(f"BGR 颜色: {bgr_color}")
        print(f"HSV 颜色: {hsv_color}\n")

def main():
    global current_frame, scale_factor
    cv2.namedWindow("实时屏幕拾色器")
    cv2.setMouseCallback("实时屏幕拾色器", pick_color)
    
    with mss.mss() as sct:
        last_time = time.time()
        while True:
            # 捕获屏幕区域
            sct_img = sct.grab(monitor)
            frame = np.array(sct_img)
            # mss 返回的是 BGRA 格式，转换为 BGR
            current_frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
            
            # 放大预览窗口显示图像
            frame_large = cv2.resize(current_frame, None, fx=scale_factor, fy=scale_factor, interpolation=cv2.INTER_LINEAR)
            cv2.imshow("实时屏幕拾色器", frame_large)
            
            current_time = time.time()
            fps = 1 / (current_time - last_time)
            last_time = current_time
            # 可选：在控制台打印 FPS 信息
            # print(f"FPS: {fps:.2f}")
            
            if cv2.waitKey(1) & 0xFF in (27, ord('q')):
                break
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
