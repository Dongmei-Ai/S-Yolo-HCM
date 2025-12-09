#!/usr/bin/env python3
"""测试 YOLOv8 和 YOLOv8-PDE 的参数数量对比"""

import torch
import torch.nn as nn
from ultralytics.nn.modules.block import Bottleneck, BottleneckDWR, C2f, C2fDWR
from ultralytics import YOLO
from pathlib import Path


def count_parameters(model):
    """计算模型的总参数数量"""
    return sum(p.numel() for p in model.parameters())


def count_trainable_parameters(model):
    """计算可训练参数数量"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def print_model_info(model, name):
    """打印模型信息"""
    total_params = count_parameters(model)
    trainable_params = count_trainable_parameters(model)
    
    print(f"\n{'='*60}")
    print(f"{name}")
    print(f"{'='*60}")
    print(f"总参数数量: {total_params:,}")
    print(f"可训练参数: {trainable_params:,}")
    
    # 打印各层参数
    print(f"\n各层参数详情:")
    print(f"{'层名':<30} {'参数数量':>15}")
    print(f"{'-'*50}")
    for name, param in model.named_parameters():
        print(f"{name:<30} {param.numel():>15,}")
    
    return total_params


def test_bottleneck_comparison():
    """对比标准 Bottleneck 和 BottleneckDWR"""
    print("\n" + "="*60)
    print("Bottleneck vs BottleneckDWR 参数对比")
    print("="*60)
    
    # 测试不同的配置
    test_configs = [
        {"c1": 128, "c2": 128, "e": 0.5, "name": "小模型 (128->128)"},
        {"c1": 256, "c2": 256, "e": 0.5, "name": "中等模型 (256->256)"},
        {"c1": 512, "c2": 512, "e": 0.5, "name": "大模型 (512->512)"},
    ]
    
    for config in test_configs:
        c1, c2, e = config["c1"], config["c2"], config["e"]
        print(f"\n{'='*60}")
        print(f"配置: {config['name']} (c1={c1}, c2={c2}, e={e})")
        print(f"{'='*60}")
        
        # 标准 Bottleneck
        bottleneck_std = Bottleneck(c1, c2, shortcut=True, e=e)
        params_std = print_model_info(bottleneck_std, "标准 Bottleneck")
        
        # BottleneckDWR
        bottleneck_dwr = BottleneckDWR(
            c1, c2, shortcut=True, e=e, 
            k=((3, 3), (3, 3), (3, 3), (3, 3), (1, 1))
        )
        params_dwr = print_model_info(bottleneck_dwr, "BottleneckDWR")
        
        # 对比
        diff = params_dwr - params_std
        diff_percent = (diff / params_std * 100) if params_std > 0 else 0
        
        print(f"\n{'='*60}")
        print(f"参数对比结果:")
        print(f"{'='*60}")
        print(f"标准 Bottleneck:     {params_std:>12,} 参数")
        print(f"BottleneckDWR:       {params_dwr:>12,} 参数")
        print(f"差异:                {diff:>12,} 参数 ({diff_percent:+.2f}%)")
        
        if diff > 0:
            print(f"⚠️  BottleneckDWR 参数更多")
        elif diff < 0:
            print(f"✅ BottleneckDWR 参数更少")
        else:
            print(f"✓   参数相同")


def test_c2f_comparison():
    """对比标准 C2f 和 C2fDWR"""
    print("\n" + "="*60)
    print("C2f vs C2fDWR 参数对比")
    print("="*60)
    
    # 测试不同的配置
    test_configs = [
        {"c1": 64, "c2": 64, "n": 1, "e": 0.5, "name": "小模型 (64->64, n=1)"},
        {"c1": 128, "c2": 128, "n": 2, "e": 0.5, "name": "中等模型 (128->128, n=2)"},
        {"c1": 256, "c2": 256, "n": 1, "e": 0.5, "name": "大模型 (256->256, n=1)"},
    ]
    
    for config in test_configs:
        c1, c2, n, e = config["c1"], config["c2"], config["n"], config["e"]
        print(f"\n{'='*60}")
        print(f"配置: {config['name']} (c1={c1}, c2={c2}, n={n}, e={e})")
        print(f"{'='*60}")
        
        # 标准 C2f
        c2f_std = C2f(c1, c2, n=n, shortcut=True, e=e)
        params_std = print_model_info(c2f_std, "标准 C2f")
        
        # C2fDWR
        c2f_dwr = C2fDWR(c1, c2, n=n, shortcut=True, e=e)
        params_dwr = print_model_info(c2f_dwr, "C2fDWR")
        
        # 对比
        diff = params_dwr - params_std
        diff_percent = (diff / params_std * 100) if params_std > 0 else 0
        
        print(f"\n{'='*60}")
        print(f"参数对比结果:")
        print(f"{'='*60}")
        print(f"标准 C2f:            {params_std:>12,} 参数")
        print(f"C2fDWR:              {params_dwr:>12,} 参数")
        print(f"差异:                {diff:>12,} 参数 ({diff_percent:+.2f}%)")
        
        if diff > 0:
            print(f"⚠️  C2fDWR 参数更多")
        elif diff < 0:
            print(f"✅ C2fDWR 参数更少")
        else:
            print(f"✓   参数相同")


def test_forward_pass():
    """测试前向传播是否正常"""
    print("\n" + "="*60)
    print("前向传播测试")
    print("="*60)
    
    c1, c2, e = 256, 256, 0.5
    batch_size, h, w = 2, 64, 64
    
    # 测试 Bottleneck
    print("\n测试标准 Bottleneck...")
    bottleneck_std = Bottleneck(c1, c2, shortcut=True, e=e)
    x_std = torch.randn(batch_size, c1, h, w)
    try:
        y_std = bottleneck_std(x_std)
        print(f"✅ 标准 Bottleneck 前向传播成功")
        print(f"   输入形状: {x_std.shape}")
        print(f"   输出形状: {y_std.shape}")
    except Exception as e:
        print(f"❌ 标准 Bottleneck 前向传播失败: {e}")
    
    # 测试 BottleneckDWR
    print("\n测试 BottleneckDWR...")
    bottleneck_dwr = BottleneckDWR(
        c1, c2, shortcut=True, e=e,
        k=((3, 3), (3, 3), (3, 3), (3, 3), (1, 1))
    )
    x_dwr = torch.randn(batch_size, c1, h, w)
    try:
        y_dwr = bottleneck_dwr(x_dwr)
        print(f"✅ BottleneckDWR 前向传播成功")
        print(f"   输入形状: {x_dwr.shape}")
        print(f"   输出形状: {y_dwr.shape}")
        
        # 检查输出形状是否一致
        if y_std.shape == y_dwr.shape:
            print(f"✅ 输出形状一致: {y_std.shape}")
        else:
            print(f"⚠️  输出形状不一致: {y_std.shape} vs {y_dwr.shape}")
    except Exception as e:
        print(f"❌ BottleneckDWR 前向传播失败: {e}")
        import traceback
        traceback.print_exc()


def test_full_model_comparison():
    """对比完整的 YOLOv8 和 YOLOv8-PDE 模型"""
    print("\n" + "="*60)
    print("完整模型参数对比: YOLOv8 vs YOLOv8-PDE")
    print("="*60)
    
    # 测试不同的模型规模
    model_scales = ['n', 's', 'm', 'l', 'x']
    
    for scale in model_scales:
        print(f"\n{'='*60}")
        print(f"测试模型规模: YOLOv8{scale} vs YOLOv8{scale}-PDE")
        print(f"{'='*60}")
        
        try:
            # 加载标准 YOLOv8
            print(f"\n加载 YOLOv8{scale}...")
            model_path_std = f'ultralytics/cfg/models/v8/yolov8{scale}.yaml'
            model_std = YOLO(model_path_std)
            
            # 获取参数数量
            params_std = sum(p.numel() for p in model_std.model.parameters())
            trainable_std = sum(p.numel() for p in model_std.model.parameters() if p.requires_grad)
            
            # 获取模型信息
            info_std = model_std.info(verbose=False, detailed=False)
            
            print(f"✅ YOLOv8{scale} 加载成功")
            print(f"   总参数: {params_std:,}")
            print(f"   可训练参数: {trainable_std:,}")
            if info_std:
                print(f"   层数: {info_std.get('layers', 'N/A')}")
                print(f"   GFLOPs: {info_std.get('GFLOPs', 'N/A')}")
            
        except Exception as e:
            print(f"❌ YOLOv8{scale} 加载失败: {e}")
            import traceback
            traceback.print_exc()
            continue
        
        try:
            # 加载 YOLOv8-PDE
            print(f"\n加载 YOLOv8{scale}-PDE...")
            model_path_pde = f'ultralytics/cfg/models/v8/yolov8{scale}-pde.yaml'
            model_pde = YOLO(model_path_pde)
            
            # 获取参数数量
            params_pde = sum(p.numel() for p in model_pde.model.parameters())
            trainable_pde = sum(p.numel() for p in model_pde.model.parameters() if p.requires_grad)
            
            # 获取模型信息
            info_pde = model_pde.info(verbose=False, detailed=False)
            
            print(f"✅ YOLOv8{scale}-PDE 加载成功")
            print(f"   总参数: {params_pde:,}")
            print(f"   可训练参数: {trainable_pde:,}")
            if info_pde:
                print(f"   层数: {info_pde.get('layers', 'N/A')}")
                print(f"   GFLOPs: {info_pde.get('GFLOPs', 'N/A')}")
            
        except Exception as e:
            print(f"❌ YOLOv8{scale}-PDE 加载失败: {e}")
            import traceback
            traceback.print_exc()
            continue
        
        # 对比结果
        diff = params_pde - params_std
        diff_percent = (diff / params_std * 100) if params_std > 0 else 0
        
        print(f"\n{'='*60}")
        print(f"参数对比结果 (YOLOv8{scale}):")
        print(f"{'='*60}")
        print(f"YOLOv8{scale}:              {params_std:>15,} 参数")
        print(f"YOLOv8{scale}-PDE:          {params_pde:>15,} 参数")
        print(f"差异:                       {diff:>15,} 参数 ({diff_percent:+.2f}%)")
        
        if diff > 0:
            print(f"⚠️  YOLOv8{scale}-PDE 参数更多 (+{diff:,}, +{diff_percent:.2f}%)")
        elif diff < 0:
            print(f"✅ YOLOv8{scale}-PDE 参数更少 ({diff:,}, {diff_percent:.2f}%)")
        else:
            print(f"✓   参数相同")
        
        # 计算参数量差异的绝对值
        abs_diff = abs(diff)
        print(f"\n参数差异绝对值: {abs_diff:,} ({abs_diff/1e6:.3f}M)")


def test_model_layers_comparison():
    """对比模型各层的参数差异"""
    print("\n" + "="*60)
    print("模型各层参数对比")
    print("="*60)
    
    scale = 'n'  # 测试 nano 版本
    
    try:
        # 加载标准 YOLOv8
        print(f"\n加载 YOLOv8{scale}...")
        model_path_std = f'ultralytics/cfg/models/v8/yolov8{scale}.yaml'
        model_std = YOLO(model_path_std)
        
        # 加载 YOLOv8-PDE
        print(f"加载 YOLOv8{scale}-PDE...")
        model_path_pde = f'ultralytics/cfg/models/v8/yolov8{scale}-pde.yaml'
        model_pde = YOLO(model_path_pde)
        
        print(f"\n{'='*60}")
        print(f"各层参数对比 (YOLOv8{scale}):")
        print(f"{'='*60}")
        print(f"{'层名':<50} {'标准':>15} {'PDE':>15} {'差异':>15}")
        print(f"{'-'*100}")
        
        # 获取所有层的参数
        std_params = {name: p.numel() for name, p in model_std.model.named_parameters()}
        pde_params = {name: p.numel() for name, p in model_pde.model.named_parameters()}
        
        # 找出所有不同的层
        all_layers = set(std_params.keys()) | set(pde_params.keys())
        
        total_diff = 0
        for layer_name in sorted(all_layers):
            std_val = std_params.get(layer_name, 0)
            pde_val = pde_params.get(layer_name, 0)
            diff = pde_val - std_val
            
            if diff != 0:  # 只显示有差异的层
                print(f"{layer_name:<50} {std_val:>15,} {pde_val:>15,} {diff:>15,}")
                total_diff += diff
        
        print(f"{'-'*100}")
        print(f"{'总计差异':<50} {'':>15} {'':>15} {total_diff:>15,}")
        
    except Exception as e:
        print(f"❌ 加载模型失败: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    print("="*60)
    print("YOLOv8 vs YOLOv8-PDE 参数对比测试工具")
    print("="*60)
    
    # 测试完整模型对比
    test_full_model_comparison()
    
    # 测试各层参数对比
    test_model_layers_comparison()
    
    # 测试 Bottleneck 对比（可选）
    print("\n" + "="*60)
    print("是否测试单个模块对比？(y/n)")
    print("="*60)
    # 取消注释下面的行来启用单个模块测试
    # test_bottleneck_comparison()
    # test_c2f_comparison()
    # test_forward_pass()
    
    print("\n" + "="*60)
    print("测试完成！")
    print("="*60)

