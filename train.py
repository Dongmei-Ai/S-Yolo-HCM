from ultralytics import YOLO

# 由于里面配置调用的不确定性，而且不好改
# 我们专门提出来一个用于存储配置的文件
# 以后的代码统一从这里取配置
from public_config import yolo_yaml, yolo_pth, dataset_yaml





# Load a model
# model = YOLO("yolo11n.yaml")  # build a new model from YAML
# model = YOLO("yolo11n.pt")  # load a pretrained model (recommended for training)
yolo_model = YOLO(yolo_yaml)  # build from YAML and transfer weights
model = yolo_model.load(yolo_pth)
# debug了半天，发现是 /server/developer/zhou/yolo/ultralytics-main/ultralytics/cfg/models/11/yolo11.yaml
# Train the model
results = model.train(data=dataset_yaml, epochs=100, imgsz=640)



