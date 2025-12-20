模型部分还不用动
看下数据是怎么自动下的
模型是怎么处理的
数据处理应该在 ultralytics/engine/trainer.py
按这个代码数据目录在/server/developer/zhou/yolo/datasets/coco8


# 前处理部分等
相关的配置文件是default_cfg_yaml
有一个 ultralytics/data/build.py 里面有一个 build_yolo_dataset 是核心
数据增强相关的内容全部都在ultralytics/data/dataset.py的build_transforms之中
关键词 前处理核心章节，参数传入部分


# 损失计算部分
损失查关键字loss, self.loss_items = self.model(batch)


# 后处理部分
后处理部分的在哪里 def postprocess(self, preds, img, orig_imgs, **kwargs): 关键词类似

NEU数据集需降低学习率，COCO及VOC都是0.01的