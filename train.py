from ultralytics import YOLO

# 由于里面配置调用的不确定性，而且不好改
# 我们专门提出来一个用于存储配置的文件
# 以后的代码统一从这里取配置
from public_config import yolo_yaml, yolo_pth, dataset_yaml





# Load a model
# model = YOLO("yolo11n.yaml")  # build a new model from YAML
# model = YOLO("yolo11n.pt")  # load a pretrained model (recommended for training)
yolo_model = YOLO(yolo_yaml)  # build from YAML and transfer weights
yolo_model = yolo_model.load(yolo_pth)
# debug了半天，发现是 /server/developer/zhou/yolo/ultralytics-main/ultralytics/cfg/models/11/yolo11.yaml
# Train the model
# YOLO11 优化建议：
# 1. 降低学习率：YOLO11 使用了 C2PSA 注意力机制，需要更小的学习率来稳定训练
# 2. 增加 batch size：如果 GPU 内存允许，建议增加到 16 或更大
# 3. 使用余弦学习率调度：有助于模型收敛
results = yolo_model.train(data=dataset_yaml, 
                            epochs=200, 
                            imgsz=640,
                            batch=4,
                            # lr0=0.005,  # 降低初始学习率（默认0.01，YOLO11建议0.005-0.008）
                            # cos_lr=True,  # 使用余弦学习率调度
                            optimizer='SGD',
                            amp=False
                            # mosaic=0.2,
                            # clahe=0.1,
                            )



# nohup python3 -u train2.py > train2.log