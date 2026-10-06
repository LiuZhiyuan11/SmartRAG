"""
YOLO11n 模型性能对比脚本
对比：PyTorch (.pt) vs ONNX (.onnx) vs TensorRT (.engine)
指标：推理延迟（平均/最小/最大）、检测结果一致性（类别、置信度、边界框）
"""
import os
import sys

# 手动添加 tensorrt_libs 目录到 DLL 搜索路径
trt_libs_path = r"C:\Users\12040\.conda\envs\omnirag\Lib\site-packages\tensorrt_libs"
if os.path.exists(trt_libs_path):
    os.add_dll_directory(trt_libs_path)
    # 同时将其加入 PATH 环境变量，作为双重保障
    os.environ["PATH"] = trt_libs_path + os.pathsep + os.environ["PATH"]


import time
import os
from pathlib import Path
from collections import Counter
import torch
from ultralytics import YOLO

# ============ 配置区 ============
IMG_PATH = r"D:\AIAgent\agent\test_image.jpeg"  # 换成你要测试的图片路径
MODELS = {
    "PyTorch (.pt)": r"D:\AIAgent\agent\yolo11n.pt",
    "ONNX (.onnx)": r"D:\AIAgent\agent\yolo11n.onnx",
    "TensorRT (.engine)": r"D:\AIAgent\agent\yolo11n.engine",
}
WARMUP = 10  # 预热次数
RUNS = 50  # 正式计时次数
DEVICE = 0 if torch.cuda.is_available() else "cpu"
CONF_THRES = 0.25  # 置信度阈值，过滤低置信度检测


def check_files():
    """检查所有模型文件和图片是否存在"""
    print("=" * 60)
    print("【文件检查】")
    if not os.path.exists(IMG_PATH):
        raise FileNotFoundError(f"测试图片不存在: {IMG_PATH}")
    print(f"  图片: {IMG_PATH} ✓")

    for name, path in MODELS.items():
        if not os.path.exists(path):
            raise FileNotFoundError(f"模型文件不存在: {path}")
        size_mb = os.path.getsize(path) / (1024 * 1024)
        print(f"  {name}: {size_mb:.1f} MB ✓")


def run_benchmark(model_name, model_path):
    """对单个模型进行计时和推理"""
    print("\n" + "=" * 60)
    print(f"【测试模型】{model_name}")

    # 加载模型
    t_load_start = time.perf_counter()
    model = YOLO(model_path, task="detect")
    load_time = time.perf_counter() - t_load_start
    print(f"  模型加载耗时: {load_time:.3f} s")

    # ---- 预热 ----
    print(f"  预热中 ({WARMUP} 次)...")
    for _ in range(WARMUP):
        model(IMG_PATH, device=DEVICE, verbose=False, conf=CONF_THRES)

    # ---- 正式计时 ----
    print(f"  计时中 ({RUNS} 次)...")
    times = []
    for _ in range(RUNS):
        t0 = time.perf_counter()
        results = model(IMG_PATH, device=DEVICE, verbose=False, conf=CONF_THRES)
        times.append((time.perf_counter() - t0) * 1000)  # 转毫秒

    times.sort()
    stats = {
        "avg_ms": sum(times) / len(times),
        "min_ms": times[0],
        "max_ms": times[-1],
        "p50_ms": times[len(times) // 2],
        "p95_ms": times[int(len(times) * 0.95)],
        "fps": 1000 / (sum(times) / len(times)),
    }

    # ---- 提取检测结果 ----
    detections = []
    for r in results:
        for box in r.boxes:
            cls_id = int(box.cls[0])
            cls_name = model.names[cls_id]
            conf = float(box.conf[0])
            xyxy = [round(float(v), 2) for v in box.xyxy[0]]
            detections.append({
                "class": cls_name,
                "conf": conf,
                "xyxy": xyxy,
            })

    return {
        "load_time": load_time,
        "stats": stats,
        "detections": detections,
    }


def compare_detections(results_dict):
    """对比三种模型的检测结果，评估一致性"""
    print("\n" + "=" * 60)
    print("【检测结果对比】")

    # 以 .pt 的结果作为基准
    baseline_name = "PyTorch (.pt)"
    baseline = results_dict[baseline_name]["detections"]

    for name, res in results_dict.items():
        dets = res["detections"]
        print(f"\n  ▶ {name}")
        print(f"    检测到 {len(dets)} 个目标")

        # 类别统计
        cls_counter = Counter([d["class"] for d in dets])
        cls_summary = ", ".join([f"{k}×{v}" for k, v in cls_counter.items()])
        print(f"    类别分布: {cls_summary}")

        # 与基准对比
        if name != baseline_name:
            baseline_classes = Counter([d["class"] for d in baseline])
            diff = cls_counter - baseline_classes
            match = cls_counter == baseline_classes

            if match:
                print(f"    ✓ 检测类别与 {baseline_name} 完全一致")
            else:
                print(f"    ⚠ 与 {baseline_name} 有差异: {dict(diff)}")

            # 对比置信度均值
            if dets and baseline:
                avg_conf = sum([d["conf"] for d in dets]) / len(dets)
                base_avg_conf = sum([d["conf"] for d in baseline]) / len(baseline)
                print(
                    f"    平均置信度: {avg_conf:.3f}  (基准: {base_avg_conf:.3f}, 差异: {avg_conf - base_avg_conf:+.3f})")


def print_summary_table(results_dict):
    """打印性能对比表"""
    print("\n" + "=" * 60)
    print("【性能对比总表】")
    print(f"{'模型':<22}{'加载(s)':<10}{'平均(ms)':<12}{'最小(ms)':<12}{'最大(ms)':<12}{'FPS':<8}")
    print("-" * 76)

    for name, res in results_dict.items():
        s = res["stats"]
        print(f"{name:<22}{res['load_time']:<10.3f}{s['avg_ms']:<12.2f}"
              f"{s['min_ms']:<12.2f}{s['max_ms']:<12.2f}{s['fps']:<8.1f}")

    # 计算加速比（以 pt 为基准）
    baseline_avg = results_dict["PyTorch (.pt)"]["stats"]["avg_ms"]
    print("\n【加速比（相对 PyTorch）】")
    for name, res in results_dict.items():
        speedup = baseline_avg / res["stats"]["avg_ms"]
        print(f"  {name}: {speedup:.2f}x")


if __name__ == "__main__":
    check_files()
    print(f"\n推理设备: {'CUDA:0' if DEVICE == 0 else 'CPU'}")

    results = {}
    for name, path in MODELS.items():
        if name == "TensorRT (.engine)" and DEVICE == "cpu":
            print(f"\n跳过 {name}: TensorRT engine 需要可用的 CUDA GPU。")
            continue
        results[name] = run_benchmark(name, path)

    compare_detections(results)
    print_summary_table(results)

    print("\n" + "=" * 60)
    print("测试完成！")