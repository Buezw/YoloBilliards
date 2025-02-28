from ultralytics import YOLO

# 这里假设你已经有模型配置文件和预训练权重文件
model = YOLO("yolo11n.yaml").load(r"yolo11n.pt")

# 使用你新建的data.yaml文件进行训练
results = model.train(data="data.yaml", epochs=60, resume=True)
