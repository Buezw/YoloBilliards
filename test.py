import pyautogui
import time

try:
    while True:
        # 获取当前鼠标的位置
        x, y = pyautogui.position()
        # 打印鼠标的位置
        print(f"Mouse Position: X = {x}, Y = {y}")
        # 每隔0.1秒检测一次
        time.sleep(0.1)
except KeyboardInterrupt:
    print("\nProgram exited.")