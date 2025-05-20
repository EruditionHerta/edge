# custom_scripts/vgg16_edge_detector.py
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F  # 导入 functional 模块用于上采样
import torchvision.models as models
import torchvision.transforms as transforms
import cv2  # OpenCV 可用于预处理/后处理

# --- 全局模型实例和预处理转换 (首次加载后缓存) ---
_vgg16_model = None
_vgg16_device = None  # 用于记录模型当前所在的设备
_vgg16_preprocess = None  # 用于存储预处理转换

# VGG16 在 ImageNet 上训练时使用的均值和标准差
VGG16_MEAN = [0.485, 0.456, 0.406]
VGG16_STD = [0.229, 0.224, 0.225]

# 我们感兴趣的 VGG16 特征层索引 (ReLU 激活之后)
# model.features 是一个 nn.Sequential 对象
# ReLU1_2 -> model.features[3]
# ReLU2_2 -> model.features[8]
# ReLU3_3 -> model.features[15]
TARGET_LAYER_INDICES = {
    'relu1_2': 3,
    'relu2_2': 8,
    'relu3_3': 15,
}

# 用于存储提取的特征图的字典
_extracted_features = {}


def get_feature_hook(layer_name):
    """定义一个 hook 函数，用于在正向传播时捕获指定层的输出。"""

    def hook(model, input, output):
        _extracted_features[layer_name] = output.detach()  # 分离张量，避免影响梯度计算

    return hook


def get_vgg16_model_and_preprocessing(device):
    """获取 VGG16 模型实例和预处理转换，如果模型未加载或设备不匹配则重新加载。"""
    global _vgg16_model, _vgg16_device, _vgg16_preprocess

    if _vgg16_model is None or _vgg16_device != device:
        print(f"[VGG16脚本] 正在加载预训练的 VGG16 模型到设备: {device}")
        model_candidate = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1).features.to(device)
        model_candidate.eval()  # 设置为评估模式

        _vgg16_model = model_candidate
        _vgg16_device = device

        _vgg16_preprocess = transforms.Compose([
            transforms.ToPILImage(),  # NumPy 数组 (H,W,C) BGR -> PIL Image
            transforms.Lambda(lambda img: img.convert('RGB')),  # 确保是 RGB
            transforms.ToTensor(),  # PIL Image -> PyTorch 张量 (C,H,W), 范围 [0,1]
            transforms.Normalize(mean=VGG16_MEAN, std=VGG16_STD)  # 标准化
        ])

        for handle in getattr(_vgg16_model, '_custom_handles', []):
            handle.remove()
        _vgg16_model._custom_handles = []  # type: ignore

        for name, idx in TARGET_LAYER_INDICES.items():
            if idx < len(_vgg16_model):
                handle = _vgg16_model[idx].register_forward_hook(get_feature_hook(name))
                _vgg16_model._custom_handles.append(handle)  # type: ignore
            else:
                print(f"[VGG16脚本] 警告: 层索引 {idx} ('{name}') 超出 VGG16 特征层范围。")

    return _vgg16_model, _vgg16_preprocess


def process_image(image_numpy_array, device):  # 参数名调整为 image_numpy_array
    """
    使用预训练的 VGG16 模型提取的特征来近似边缘。
    此函数符合外部自定义算法模块的接口要求。

    参数:
        image_numpy_array (np.ndarray): 输入的图像，期望是 BGR 顺序的 NumPy 数组 (H, W, C)。
        device (torch.device): 模型运行的设备 (例如, torch.device('cuda:0') 或 torch.device('cpu'))。

    返回:
        np.ndarray: 二值边缘图 (H, W), 值为 0 或 255, 类型为 np.uint8。
    """
    print(f"[VGG16脚本] 接收到图像，形状: {image_numpy_array.shape}, 类型: {image_numpy_array.dtype}")
    print(f"[VGG16脚本] 使用设备: {device}")

    # 确认输入是 BGR 格式的 NumPy 数组
    if image_numpy_array.ndim != 3 or image_numpy_array.shape[2] != 3:
        print("[VGG16脚本] 错误: 输入图像期望是3通道 (BGR) NumPy 数组。")
        # 返回一个全黑的图像作为错误指示
        return np.zeros((image_numpy_array.shape[0], image_numpy_array.shape[1]), dtype=np.uint8)

    model, preprocess = get_vgg16_model_and_preprocessing(device)

    original_height, original_width = image_numpy_array.shape[:2]

    # 1. 预处理图像
    # image_numpy_array 已经是 BGR HWC NumPy 数组
    input_tensor = preprocess(image_numpy_array).unsqueeze(0).to(device)
    print(f"[VGG16脚本] 输入张量形状: {input_tensor.shape}")

    # 2. 模型推理 (正向传播)
    _extracted_features.clear()
    with torch.no_grad():
        model(input_tensor)

    # 3. 特征处理与融合
    fused_response_map = None

    if not _extracted_features:
        print("[VGG16脚本] 错误: 未能从 VGG16 提取任何特征。")
        return np.zeros((original_height, original_width), dtype=np.uint8)

    for layer_name, feature_map in _extracted_features.items():
        print(f"[VGG16脚本] 层 '{layer_name}' 的特征图形状: {feature_map.shape}")
        upsampled_feature = F.interpolate(feature_map,
                                          size=(original_height, original_width),
                                          mode='bilinear',
                                          align_corners=False)
        response = torch.mean(upsampled_feature, dim=1, keepdim=True)

        if fused_response_map is None:
            fused_response_map = response
        else:
            fused_response_map += response

    if fused_response_map is None:
        print("[VGG16脚本] 错误: 未能生成融合后的响应图。")
        return np.zeros((original_height, original_width), dtype=np.uint8)

    edge_strength_map = fused_response_map.squeeze().cpu().numpy()
    print(f"[VGG16脚本] 融合后的边缘强度图形状: {edge_strength_map.shape}")

    # 4. 后处理：归一化和阈值化
    if np.max(edge_strength_map) > np.min(edge_strength_map):
        edge_strength_map = (edge_strength_map - np.min(edge_strength_map)) / \
                            (np.max(edge_strength_map) - np.min(edge_strength_map))
    else:
        edge_strength_map = np.zeros_like(edge_strength_map)

    threshold = 0.3  # 这个阈值可能需要根据实际效果调整
    binary_edge_map_01 = (edge_strength_map > threshold).astype(np.uint8)
    binary_edge_map_0255 = (binary_edge_map_01 * 255).astype(np.uint8)

    print(f"[VGG16脚本] 输出二值图形状: {binary_edge_map_0255.shape}, 最大值: {np.max(binary_edge_map_0255)}")
    return binary_edge_map_0255


if __name__ == '__main__':
    # 独立测试此脚本的示例
    dummy_image_bgr = np.random.randint(0, 256, (224, 224, 3), dtype=np.uint8)

    print("\n使用 CPU 测试:")
    cpu_device = torch.device('cpu')
    edge_map_cpu = process_image(dummy_image_bgr.copy(), cpu_device)
    if edge_map_cpu is not None:
        print(f"CPU 输出形状: {edge_map_cpu.shape}, 类型: {edge_map_cpu.dtype}")
        # cv2.imshow("VGG16 Edges CPU", edge_map_cpu)

    if torch.cuda.is_available():
        print("\n使用 GPU 测试:")
        gpu_device = torch.device('cuda')
        edge_map_gpu = process_image(dummy_image_bgr.copy(), gpu_device)
        if edge_map_gpu is not None:
            print(f"GPU 输出形状: {edge_map_gpu.shape}, 类型: {edge_map_gpu.dtype}")
            # cv2.imshow("VGG16 Edges GPU", edge_map_gpu)
    else:
        print("\nGPU 不可用，跳过 GPU 测试。")

    # cv2.waitKey(0)
    # cv2.destroyAllWindows()