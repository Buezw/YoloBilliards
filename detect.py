import cv2
import numpy as np
import mss
from ultralytics import YOLO
import time
import pyautogui
import math

# 加载预训练模型，请确保路径正确
model = YOLO(r"train10\weights\last.pt")

# 定义屏幕捕获区域（根据实际分辨率调整）
monitor = {"top": 240, "left": 640, "width": 1917 - 640, "height": 1160 - 240}

# 定义预览窗口大小（主动缩放图像到该尺寸）
preview_width = 1200
preview_height = 700
scale_x = preview_width / monitor['width']
scale_y = preview_height / monitor['height']

# 定义颜色范围（HSV 范围）
colors_ranges = {
    "red":    ((0, 200, 100), (10, 255, 255)),
    "green":  ((35, 150, 100), (55, 255, 255)),
    "yellow": ((20, 200, 200), (30, 255, 255)),
    "orange": ((15, 200, 200), (25, 255, 255)),
    "blue":   ((115, 200, 180), (127, 255, 255)),
    "purple": ((130, 50, 100), (150, 255, 255)),
    "brown":  ((5, 150, 70), (15, 255, 150))
}

def identify_color(crop_img):
    hsv_img = cv2.cvtColor(crop_img, cv2.COLOR_RGB2HSV)
    for color_name, (lower, upper) in colors_ranges.items():
        mask = cv2.inRange(hsv_img, np.array(lower), np.array(upper))
        if cv2.countNonZero(mask) > 0:
            return color_name
    return "unknown"

def compute_iou(box1, box2):
    x1, y1, x2, y2 = box1
    xx1, yy1, xx2, yy2 = box2
    inter_x1 = max(x1, xx1)
    inter_y1 = max(y1, yy1)
    inter_x2 = min(x2, xx2)
    inter_y2 = min(y2, yy2)
    inter_area = max(0, inter_x2 - inter_x1) * max(0, inter_y2 - inter_y1)
    area1 = (x2 - x1) * (y2 - y1)
    area2 = (xx2 - xx1) * (yy2 - yy1)
    iou = inter_area / (area1 + area2 - inter_area + 1e-6)
    return iou

def compute_ball_radius(_):
    """固定球的半径为21.5像素（对应直径43像素）"""
    return 21.5

def distance_point_to_line(P, A, B):
    A3 = np.append(np.array(A, dtype=float), 0)
    B3 = np.append(np.array(B, dtype=float), 0)
    P3 = np.append(np.array(P, dtype=float), 0)
    cross = np.cross(B3 - A3, P3 - A3)
    return np.linalg.norm(cross) / (np.linalg.norm(B3 - A3) + 1e-6)

def is_point_on_segment(P, A, B):
    A = np.array(A, dtype=float)
    B = np.array(B, dtype=float)
    P = np.array(P, dtype=float)
    AB = B - A
    AP = P - A
    t = np.dot(AP, AB) / (np.dot(AB, AB) + 1e-6)
    return 0 <= t <= 1

def is_path_clear(A, B, obstacles, moving_ball_radius):
    for obs in obstacles:
        obs_center = obs['center']
        obs_radius = obs['radius']
        clearance = moving_ball_radius + obs_radius
        d = distance_point_to_line(obs_center, A, B)
        if d < clearance and is_point_on_segment(obs_center, A, B):
            return False
    return True

def normalize(v):
    norm = np.linalg.norm(v)
    return v if norm == 0 else (v / norm)

# 全局变量：记录最新检测到的所有球、目标球、方案及是否已执行判断
last_detections = []
selected_target = None
judgement_executed = False
shot_executed = False
shot_plans = []         # 保存所有可执行方案
selected_shot_plan = None  # 用户选定的方案
mouse_moved = False      # 标记是否已经移动过鼠标

def select_target_or_shot(event, x, y, flags, param):
    """
    鼠标回调函数：
    将鼠标点击的预览窗口坐标转换回原始截图区域的坐标，
    再进行目标球或目标袋口的选择处理。
    """
    global selected_target, last_detections, judgement_executed, shot_plans, selected_shot_plan, mouse_moved

    orig_x = int(x / scale_x)
    orig_y = int(y / scale_y)

    if event == cv2.EVENT_LBUTTONDOWN:
        # 若存在击打方案，先检测是否点击袋口
        if shot_plans:
            for plan in shot_plans:
                pocket_coord = plan['pocket']
                d = math.hypot(pocket_coord[0] - orig_x, pocket_coord[1] - orig_y)
                if d < 30:
                    selected_shot_plan = plan
                    print(f"选定方案：袋口【{plan['pocket_name']}】")
                    # 目标击球位置
                    abs_target_contact_point = (
                        monitor['left'] + plan['target_contact_point'][0],
                        monitor['top'] + plan['target_contact_point'][1]
                    )
                    print("预测的目标击球位置（绝对屏幕坐标）：", abs_target_contact_point)
                    # 移动鼠标并自动点击一次左键
                    pyautogui.moveTo(abs_target_contact_point[0],
                                       abs_target_contact_point[1])
                    time.sleep(0.3)
                    pyautogui.click()
                    pyautogui.click()   # 自动点击一次左键
                    mouse_moved = True
                    return

        # 如果没有点击袋口，则尝试选定目标球
        min_distance = float('inf')
        chosen = None
        for det in last_detections:
            center = det.get('center')
            if center is not None:
                d = math.hypot(center[0] - orig_x, center[1] - orig_y)
                if d < min_distance:
                    min_distance = d
                    chosen = det
        if min_distance < 30:
            selected_target = chosen
            selected_shot_plan = None
            shot_plans.clear()
            judgement_executed = False
            mouse_moved = False  # 重选目标后重置鼠标移动标志
            print(f"已选定目标球：{selected_target['label']}，中心坐标：{selected_target['center']}")

def main():
    global last_detections, selected_target, judgement_executed, shot_executed, shot_plans, selected_shot_plan, mouse_moved
    fixed_size = 43  # 固定尺寸设为43（直径43，即球半径21.5）
    half1 = fixed_size // 2      # 43//2 = 21
    half2 = fixed_size - half1   # 43 - 21 = 22

    cv2.namedWindow("Detect", cv2.WINDOW_NORMAL)
    cv2.moveWindow("Detect", 0, 0)
    cv2.setMouseCallback("Detect", select_target_or_shot)

    # 定义袋口名称与坐标（均为原始截图区域坐标）
    pockets = {
        "LU": (60, 101),
        "MU": (640, 82),
        "RI": (1216, 101),
        "LD": (60, 672),
        "MD": (640, 690),
        "RD": (1216, 672)
    }

    with mss.mss() as sct:
        while True:
            sct_img = sct.grab(monitor)
            frame = np.array(sct_img)
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGRA2RGB)
            results = model(frame_rgb, verbose=False)
            
            detections = []
            for i, box in enumerate(results[0].boxes.xyxy):
                # 取检测框中心，并以固定尺寸构造43×43的边界框
                orig_box = list(map(int, box.tolist()))
                center = ((orig_box[0] + orig_box[2]) // 2,
                          (orig_box[1] + orig_box[3]) // 2)
                bbox = (center[0] - half1, center[1] - half1,
                        center[0] + half2, center[1] + half2)
                cls = int(results[0].boxes.cls[i])
                label = model.names[cls] if hasattr(model, "names") else str(cls)
                conf = float(results[0].boxes.conf[i])
                radius = compute_ball_radius(bbox)
                detections.append({
                    'bbox': bbox,
                    'label': label,
                    'conf': conf,
                    'center': center,
                    'radius': radius
                })
            
            # 后处理：若同一目标同时被检测为 White 与 Odd，则保留 White
            final_detections = detections.copy()
            iou_threshold = 0.5
            for det_white in detections:
                if det_white['label'] == 'White':
                    for det in detections:
                        if det['label'] == 'Odd' and det in final_detections:
                            if compute_iou(det_white['bbox'], det['bbox']) > iou_threshold:
                                final_detections.remove(det)
            
            # 绘制检测框和颜色信息
            for det in final_detections:
                x1, y1, x2, y2 = det['bbox']
                label = det['label']
                conf = det['conf']
                crop_img = frame_rgb[max(0, y1):y2, max(0, x1):x2]
                color = identify_color(crop_img)
                final_label = f"{label} ({color})"
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, f"{final_label} {conf:.2f}", (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            
            last_detections = final_detections.copy()
            
            # 绘制袋口标记
            for pocket_name, (bx, by) in pockets.items():
                cv2.circle(frame, (bx, by), 8, (0, 0, 255), -1)
                cv2.putText(frame, f"{pocket_name}", (bx - 20, by - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
            
            # —— 桌球模拟逻辑 ——
            white_det = None
            for det in final_detections:
                if det['label'].lower() == 'white':
                    white_det = det
                    break
            
            if white_det is not None and selected_target is not None and not judgement_executed:
                white_center = white_det['center']
                white_radius = white_det['radius']  # 此时白球半径为21.5
                target = selected_target  # 用户选定的目标球
                target_center = target['center']
                target_radius = target['radius']  # 目标球半径也设为21.5
                
                shot_plans.clear()
                # 遍历每个袋口，计算击球方案
                for pocket_name, pocket_coord in pockets.items():
                    v_tp = np.array(pocket_coord) - np.array(target_center)
                    if np.linalg.norm(v_tp) == 0:
                        continue
                    v_tp_norm = normalize(v_tp)
                    # 目标接触点：目标球中心沿远离袋口方向 43 像素（球直径43）
                    target_contact_point = np.array(target_center) - 43 * v_tp_norm
                    # 白球击球位置：在 target_contact_point 再沿同一方向后退白球半径（21.5像素）
                    white_hit_center = target_contact_point - white_radius * v_tp_norm

                    obstacles_w = [det for det in final_detections if det['center'] not in (white_center, target_center)]
                    if not is_path_clear(white_center, tuple(white_hit_center.astype(int)), obstacles_w, white_radius):
                        continue
                    obstacles_tp = [det for det in final_detections if det['center'] != target_center]
                    if not is_path_clear(target_center, pocket_coord, obstacles_tp, target_radius):
                        continue

                    shot_plans.append({
                        'white_hit_center': tuple(white_hit_center.astype(int)),
                        'target_contact_point': tuple(target_contact_point.astype(int)),
                        'pocket': pocket_coord,
                        'pocket_name': pocket_name,
                        'white_center': white_center,
                        'target_center': target_center
                    })
                    cv2.line(frame, white_center, tuple(white_hit_center.astype(int)), (255, 0, 0), 2)
                    cv2.line(frame, target_center, pocket_coord, (255, 0, 0), 2)
                    cv2.circle(frame, tuple(white_hit_center.astype(int)), 6, (255, 255, 0), -1)

                if shot_plans:
                    plans_str = " ".join([f"袋口【{plan['pocket_name']}】" for plan in shot_plans])
                    print("可击打方案：", plans_str)
                else:
                    print("目标球到所有袋口的路径均不畅通，无法击打。")
                
                judgement_executed = True
            
            # 选定方案后：画红色轨迹 + 画实际43×43像素的圆（半径取 int(round(21.5))）
            if selected_shot_plan is not None:
                cv2.line(frame, selected_shot_plan['white_center'],
                         selected_shot_plan['target_contact_point'], (0, 0, 255), 2)
                cv2.line(frame, selected_shot_plan['target_center'],
                         selected_shot_plan['pocket'], (0, 0, 255), 2)
                
                landing = selected_shot_plan['target_contact_point']
                cv2.circle(frame, landing, int(round(21.5)), (0, 255, 255), 2)
            
            scaled_frame = cv2.resize(frame, (preview_width, preview_height))
            cv2.imshow("Detect", scaled_frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('r'):
                shot_executed = False
                selected_target = None
                judgement_executed = False
                shot_plans.clear()
                selected_shot_plan = None
                mouse_moved = False
            if key == ord('q') or key == 27:
                break
        
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
