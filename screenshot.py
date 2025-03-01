import time
import mss
import mss.tools

def capture_screen():
    with mss.mss() as sct:
        # 定义捕捉区域
        monitor = {
            "top": 240,
            "left": 640,
            "width": 1917 - 640,
            "height": 1160 - 240
        }
        while True:
            screenshot = sct.grab(monitor)
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            filename = f"screenshot_{timestamp}.png"
            mss.tools.to_png(screenshot.rgb, screenshot.size, output=filename)
            print(f"已保存: {filename}")
            time.sleep(5)

if __name__ == "__main__":
    capture_screen()
