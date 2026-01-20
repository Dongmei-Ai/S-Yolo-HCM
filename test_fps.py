#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
简单的FPS测试脚本
用于测试YOLO模型的推理速度
"""

import time
import torch
import numpy as np
from pathlib import Path
from ultralytics import YOLO


def test_fps(model_path, num_runs=100, imgsz=640, device='cuda', warmup=10):
    """
    测试模型的FPS
    
    Args:
        model_path: 模型文件路径 (如 'runs/detect/train/weights/best.pt')
        num_runs: 测试运行次数，默认100次
        imgsz: 输入图像尺寸，默认640
        device: 设备类型，'cuda' 或 'cpu'
        warmup: 预热次数，默认10次
    
    Returns:
        dict: 包含FPS统计信息的字典
    """
    print(f"正在加载模型: {model_path}")
    model = YOLO(model_path)
    
    # 检查设备
    if device == 'cuda' and not torch.cuda.is_available():
        print("警告: CUDA不可用，使用CPU")
        device = 'cpu'
    
    # 创建随机测试图像 (模拟真实输入)
    test_image = np.random.randint(0, 255, (imgsz, imgsz, 3), dtype=np.uint8)
    
    print(f"\n设备: {device}")
    print(f"图像尺寸: {imgsz}x{imgsz}")
    print(f"预热次数: {warmup}")
    print(f"测试次数: {num_runs}")
    print("\n开始预热...")
    
    # 预热阶段
    for _ in range(warmup):
        _ = model(test_image, imgsz=imgsz, device=device, verbose=False)
    
    # 同步GPU（如果使用CUDA）
    if device == 'cuda':
        torch.cuda.synchronize()
    
    print("预热完成，开始正式测试...\n")
    
    # 正式测试
    times = []
    for i in range(num_runs):
        if device == 'cuda':
            torch.cuda.synchronize()
        
        start_time = time.time()
        _ = model(test_image, imgsz=imgsz, device=device, verbose=False)
        
        if device == 'cuda':
            torch.cuda.synchronize()
        
        end_time = time.time()
        elapsed = (end_time - start_time) * 1000  # 转换为毫秒
        times.append(elapsed)
        
        if (i + 1) % 10 == 0:
            print(f"已完成 {i+1}/{num_runs} 次测试...")
    
    # 计算统计信息
    times = np.array(times)
    mean_time = np.mean(times)
    std_time = np.std(times)
    min_time = np.min(times)
    max_time = np.max(times)
    median_time = np.median(times)
    
    # 计算FPS
    mean_fps = 1000 / mean_time
    max_fps = 1000 / min_time
    min_fps = 1000 / max_time
    median_fps = 1000 / median_time
    
    results = {
        'mean_time_ms': mean_time,
        'std_time_ms': std_time,
        'min_time_ms': min_time,
        'max_time_ms': max_time,
        'median_time_ms': median_time,
        'mean_fps': mean_fps,
        'max_fps': max_fps,
        'min_fps': min_fps,
        'median_fps': median_fps,
    }
    
    # 打印结果
    print("\n" + "="*60)
    print("FPS测试结果")
    print("="*60)
    print(f"平均推理时间: {mean_time:.2f} ms")
    print(f"标准差:       {std_time:.2f} ms")
    print(f"最小时间:     {min_time:.2f} ms")
    print(f"最大时间:     {max_time:.2f} ms")
    print(f"中位数时间:   {median_time:.2f} ms")
    print("-"*60)
    print(f"平均FPS:      {mean_fps:.2f} FPS")
    print(f"最大FPS:      {max_fps:.2f} FPS")
    print(f"最小FPS:      {min_fps:.2f} FPS")
    print(f"中位数FPS:    {median_fps:.2f} FPS")
    print("="*60)
    
    return results


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='YOLO模型FPS测试工具')
    parser.add_argument('--model', type=str, default='best.pt',
                        help='模型文件路径 (默认: runs/detect/train/weights/best.pt)')
    parser.add_argument('--runs', type=int, default=100,
                        help='测试运行次数 (默认: 100)')
    parser.add_argument('--imgsz', type=int, default=640,
                        help='输入图像尺寸 (默认: 640)')
    parser.add_argument('--device', type=str, default='cuda',
                        choices=['cuda', 'cpu'],
                        help='设备类型 (默认: cuda)')
    parser.add_argument('--warmup', type=int, default=10,
                        help='预热次数 (默认: 10)')
    
    args = parser.parse_args()
    
    # 检查模型文件是否存在
    model_path = Path(args.model)
    if not model_path.exists():
        print(f"错误: 模型文件不存在: {args.model}")
        print("\n提示: 请指定正确的模型路径，例如:")
        print("  python test_fps.py --model runs/detect/train/weights/best.pt")
        exit(1)
    
    # 运行测试
    test_fps(
        model_path=str(model_path),
        num_runs=args.runs,
        imgsz=args.imgsz,
        device=args.device,
        warmup=args.warmup
    )

