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
    return inter_area / (area1 + area2 - inter_area + 1e-6)

def compute_ball_radius(_):
    """固定球的半径为21.5像素（对应直径43像素）"""
    return 22

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

def angle_between_vectors(vec1, vec2):
    """计算两个向量的夹角（单位：度）"""
    dot = np.dot(vec1, vec2)
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)
    cos_theta = dot / (norm1 * norm2 + 1e-6)
    cos_theta = np.clip(cos_theta, -1.0, 1.0)
    angle = np.arccos(cos_theta)
    return np.degrees(angle)

# 全局变量：原有鼠标指定流程保留
selected_target = None
judgement_executed = False
shot_executed = False
shot_plans = []         
selected_shot_plan = None  
mouse_moved = False        
final_detections_global = []  # 保存最新检测结果

# 新增全局变量，记录白球上一次的位置
last_white_center = None
white_move_threshold = 5  # 像素阈值

def select_target_or_shot(event, x, y, flags, param):
    """
    鼠标回调函数：
    将鼠标点击的预览窗口坐标转换回原始截图区域坐标，
    若点击的是目标球，则将其设为 selected_target；
    若点击的是袋口，则执行原有击打操作。
    """
    global selected_target, shot_plans, selected_shot_plan, judgement_executed, mouse_moved

    orig_x = int(x / scale_x)
    orig_y = int(y / scale_y)

    if event == cv2.EVENT_LBUTTONDOWN:
        # 若已有击打方案，先检测是否点击袋口
        if shot_plans:
            for plan in shot_plans:
                pocket_coord = plan['pocket']
                d = math.hypot(pocket_coord[0] - orig_x, pocket_coord[1] - orig_y)
                if d < 30:
                    selected_shot_plan = plan
                    print(f"选定方案：袋口【{plan['pocket_name']}】")
                    abs_target_contact_point = (
                        monitor['left'] + plan['target_contact_point'][0],
                        monitor['top'] + plan['target_contact_point'][1]
                    )
                    print("预测的目标击球位置（绝对屏幕坐标）：", abs_target_contact_point)
                    pyautogui.moveTo(abs_target_contact_point[0], abs_target_contact_point[1])
                    time.sleep(0.3)
                    pyautogui.click()
                    pyautogui.click()
                    mouse_moved = True
                    return

        # 未点击袋口，则认为是选定目标球
        min_distance = float('inf')
        chosen = None
        for det in final_detections_global:
            center = det.get('center')
            if center is not None:
                d = math.hypot(center[0] - orig_x, center[1] - orig_y)
                if d < min_distance:
                    min_distance = d
                    chosen = det
        if min_distance < 30:
            selected_target = chosen
            shot_plans.clear()
            judgement_executed = False
            mouse_moved = False
            print(f"已选定目标球：{selected_target['label']}，中心坐标：{selected_target['center']}")

# 用于鼠标点击退出程序
def mouse_exit(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        cv2.destroyAllWindows()
        exit()

def main():
    global final_detections_global, selected_target, judgement_executed, shot_executed, shot_plans, selected_shot_plan, mouse_moved, last_white_center

    # 程序开始时等待输入：even 或 odd
    ball_type = input("请输入球类型 (even/odd): ").strip().lower()
    if ball_type not in ['even', 'odd']:
        print("输入错误，默认为 odd")
        ball_type = 'odd'

    fixed_size = 44      # 球直径43像素，半径21.5
    half1 = fixed_size // 2
    half2 = fixed_size - half1

    cv2.namedWindow("Detect", cv2.WINDOW_NORMAL)
    cv2.moveWindow("Detect", 0, 0)
    cv2.setMouseCallback("Detect", select_target_or_shot)

    # 定义袋口名称与坐标（原始截图区域坐标）
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
            
            # 后处理：若同一目标同时检测为 White 与 Odd/Even，则保留 White
            final_detections = detections.copy()
            iou_threshold = 0.5
            for det_white in detections:
                if det_white['label'] == 'White':
                    for det in detections:
                        if det['label'] in ['Odd', 'Even'] and det in final_detections:
                            if compute_iou(det_white['bbox'], det['bbox']) > iou_threshold:
                                final_detections.remove(det)
            
            final_detections_global = final_detections.copy()

            # 绘制检测框及颜色信息
            for det in final_detections:
                center = det['center']
                conf = det['conf']
                crop_img = frame_rgb[
                    max(0, center[1] - int(det['radius'])) : center[1] + int(det['radius']),
                    max(0, center[0] - int(det['radius'])) : center[0] + int(det['radius'])
                ]
                color = identify_color(crop_img)
                final_label = f"{det['label']} ({color})"
                cv2.circle(frame, center, int(det['radius']), (0, 255, 0), 2)
                cv2.putText(frame, f"{final_label} {conf:.2f}", (center[0]-20, center[1]-10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

            # 绘制袋口标记
            for pocket_name, (bx, by) in pockets.items():
                cv2.circle(frame, (bx, by), 8, (0, 0, 255), -1)
                cv2.putText(frame, pocket_name, (bx - 20, by - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
            
            # 寻找白球（只取第一个白球）
            white_det = None
            for det in final_detections:
                if det['label'].lower() == 'white':
                    white_det = det
                    break

            if white_det is not None:
                white_center = white_det['center']
                white_radius = white_det['radius']  # 固定为21.5

                # 如果已选定袋口方案，且白球移动明显则重置状态重新预测方案
                if selected_shot_plan is not None:
                    if last_white_center is None:
                        last_white_center = white_center
                    distance = np.linalg.norm(np.array(white_center) - np.array(last_white_center))
                    if distance >= white_move_threshold:
                        print("白球移动，重新预测方案")
                        selected_target = None
                        selected_shot_plan = None
                        shot_plans.clear()
                        last_white_center = white_center

                # 当选定了目标球时，计算对应方案
                if selected_target is not None:
                    if last_white_center is None:
                        last_white_center = white_center
                    distance = np.linalg.norm(np.array(white_center) - np.array(last_white_center))
                    if distance >= white_move_threshold or not shot_plans:
                        last_white_center = white_center
                        target = selected_target
                        target_center = target['center']
                        target_radius = target['radius']
                        shot_plans.clear()
                        for pocket_name, pocket_coord in pockets.items():
                            v_tp = np.array(pocket_coord) - np.array(target_center)
                            if np.linalg.norm(v_tp) == 0:
                                continue
                            v_tp_norm = normalize(v_tp)
                            target_contact_point = np.array(target_center) - 44 * v_tp_norm
                            white_hit_center = target_contact_point - white_radius * v_tp_norm
                            obstacles_w = [d for d in final_detections if d['center'] not in (white_center, target_center)]
                            if not is_path_clear(white_center, tuple(white_hit_center.astype(int)), obstacles_w, white_radius):
                                continue
                            obstacles_tp = [d for d in final_detections if d['center'] != target_center]
                            if not is_path_clear(target_center, pocket_coord, obstacles_tp, target_radius):
                                continue
                            v1 = np.array(target_contact_point) - np.array(white_center)
                            v2 = np.array(pocket_coord) - np.array(target_center)
                            angle_deg = angle_between_vectors(v1, v2)
                            if angle_deg > 120:
                                continue
                            shot_plans.append({
                                'white_hit_center': tuple(white_hit_center.astype(int)),
                                'target_contact_point': tuple(target_contact_point.astype(int)),
                                'pocket': pocket_coord,
                                'pocket_name': pocket_name,
                                'white_center': white_center,
                                'target_center': target_center,
                                'angle_deg': angle_deg
                            })
                    # 若白球未明显移动且已有方案，则继续使用当前方案

                else:
                    # 未指定目标球时，根据输入的球类型遍历所有候选球依次计算方案
                    shot_plans_by_target = []  
                    for det in final_detections:
                        if det['label'].lower() == ball_type:
                            target = det
                            target_center = target['center']
                            target_radius = target['radius']
                            plans = []
                            for pocket_name, pocket_coord in pockets.items():
                                v_tp = np.array(pocket_coord) - np.array(target_center)
                                if np.linalg.norm(v_tp) == 0:
                                    continue
                                v_tp_norm = normalize(v_tp)
                                target_contact_point = np.array(target_center) - 44 * v_tp_norm
                                white_hit_center = target_contact_point - white_radius * v_tp_norm
                                obstacles_w = [d for d in final_detections if d['center'] not in (white_center, target_center)]
                                if not is_path_clear(white_center, tuple(white_hit_center.astype(int)), obstacles_w, white_radius):
                                    continue
                                obstacles_tp = [d for d in final_detections if d['center'] != target_center]
                                if not is_path_clear(target_center, pocket_coord, obstacles_tp, target_radius):
                                    continue
                                v1 = np.array(target_contact_point) - np.array(white_center)
                                v2 = np.array(pocket_coord) - np.array(target_center)
                                angle_deg = angle_between_vectors(v1, v2)
                                # odd 球过滤角度为120度，even 球也可根据需要调整此条件
                                if ball_type == 'odd' and angle_deg > 120:
                                    continue
                                if ball_type == 'even' and angle_deg > 120:
                                    continue
                                plans.append({
                                    'white_hit_center': tuple(white_hit_center.astype(int)),
                                    'target_contact_point': tuple(target_contact_point.astype(int)),
                                    'pocket': pocket_coord,
                                    'pocket_name': pocket_name,
                                    'white_center': white_center,
                                    'target_center': target_center,
                                    'angle_deg': angle_deg
                                })
                            if plans:
                                shot_plans_by_target.append({
                                    'target': target,
                                    'plans': plans
                                })
                    shot_plans = []
                    for group in shot_plans_by_target:
                        shot_plans.extend(group['plans'])
                        cv2.circle(frame, group['target']['center'], int(group['target']['radius']), (255, 255, 0), 2)
            
            # 绘制所有方案（红色直线、目标接触点处画圆）
            if shot_plans:
                for plan in shot_plans:
                    cv2.line(frame, plan['white_center'],
                             plan['target_contact_point'], (0, 0, 255), 2)
                    cv2.line(frame, plan['target_center'],
                             plan['pocket'], (0, 0, 255), 2)
                    landing = plan['target_contact_point']
                    cv2.circle(frame, landing, int(round(22)), (0, 255, 255), 2)
            
            scaled_frame = cv2.resize(frame, (preview_width, preview_height))
            cv2.imshow("Detect", scaled_frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('r'):
                selected_target = None
                shot_plans.clear()
                selected_shot_plan = None
                judgement_executed = False
                mouse_moved = False
                last_white_center = None
            if key == ord('q') or key == 27:
                break
        
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
