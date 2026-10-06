import logging
import time
from ultralytics import YOLO
from collections import Counter

logger = logging.getLogger(__name__)

class VisionService:
    def __init__(self, model_name: str = "yolo11n.engine"):
        # 单例加载模型，避免每次调用都重新加载
        self.model = YOLO(model_name, task="detect")
        logger.info(f"YOLO 模型 {model_name} 加载完成")

    def detect(self, image_path: str) ->str:
        """执行目标检测，返回格式化的文本结果"""
        started_at = time.perf_counter()
        logger.info("YOLO 推理开始: image_path=%s device=0 conf=0.25", image_path)
        results = self.model(image_path, device=0, verbose=False, conf=0.25)
        detections = []
        for r in results:
            for box in r.boxes:
                cls_name = self.model.names[int(box.cls)]
                conf = float(box.conf)
                if conf > 0.25:
                    detections.append(f"检测到: {cls_name}, 置信度: {conf:.2f}")
        if not detections:
            logger.info(
                "YOLO 推理完成: 未检测到目标，耗时 %.3fs",
                time.perf_counter() - started_at,
            )
            return "没有检测到目标。"

        # 去重统计
        counter = Counter([detection.split(",")[0] for detection in detections])
        summary = "、".join([f"{name} × {count}" for name, count in counter.items()])

        result = f"检测到以下物体:{summary}"
        logger.info(
            "YOLO 推理完成: detections=%d summary=%s elapsed=%.3fs",
            len(detections),
            summary,
            time.perf_counter() - started_at,
        )
        return result

# 全局单例
vision_service = VisionService()