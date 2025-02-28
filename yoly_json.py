import json
import os

def labelme_to_yolo(label_me_json_file, cls2id_dict):
    with open(label_me_json_file, mode='r', encoding='UTF-8') as f:
        label_me_json = json.load(f)
    shapes = label_me_json['shapes']
    img_width, img_height = label_me_json['imageWidth'], label_me_json['imageHeight']

    labels = []
    for s in shapes:
        s_type = s['shape_type'].lower()
        # 只处理 rectangle 和 polygon 类型
        if s_type in ['rectangle', 'polygon']:
            pts = s['points']
            # 对多边形同样计算所有点的最小和最大值作为边界框
            xs = [p[0] for p in pts]
            ys = [p[1] for p in pts]
            x1, x2 = min(xs), max(xs)
            y1, y2 = min(ys), max(ys)
            # 计算中心点、宽度和高度（归一化）
            x_center = (x1 + x2) / 2 / img_width
            y_center = (y1 + y2) / 2 / img_height
            w = (x2 - x1) / img_width
            h = (y2 - y1) / img_height
            cid = cls2id_dict[s['label']]
            labels.append(f'{cid} {x_center} {y_center} {w} {h}')
    return labels

def write_label2txt(save_txt_path, label_list):
    with open(save_txt_path, "w", encoding="UTF-8") as f:
        for label in label_list:
            f.write(label + "\n")

if __name__ == '__main__':
    # 原始图片文件夹路径
    img_dir = r"E:\OneDrive\Gits\YoloBilliards\Screenshot\images"
    # 原始JSON标签文件夹路径
    json_dir = r"E:\OneDrive\Gits\YoloBilliards\Screenshot\json"
    # 生成保存TXT文件夹路径
    save_dir = r"E:\OneDrive\Gits\YoloBilliards\Screenshot"
    # 类别和序号的映射字典
    cls2id_dict = {"White": "0", "Black": "1", "Odd": "2", "Even": "3"}

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # 只处理后缀为 .json 的文件
    for json_name in os.listdir(json_dir):
        if not json_name.lower().endswith('.json'):
            continue
        json_path = os.path.join(json_dir, json_name)
        txt_name = os.path.splitext(json_name)[0] + ".txt"
        save_txt_path = os.path.join(save_dir, txt_name)
        labels = labelme_to_yolo(json_path, cls2id_dict)
        write_label2txt(save_txt_path, labels)
        print(f"已转换: {json_name} -> {txt_name}")
