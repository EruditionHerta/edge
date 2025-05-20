# core/evaluation.py
import numpy as np
from skimage.metrics import structural_similarity as ssim  # 导入 SSIM
from skimage.metrics import peak_signal_noise_ratio as psnr  # 导入 PSNR


def calculate_pixel_metrics(predicted_edge_map, ground_truth_map):
    """
    计算基于像素的边缘检测评价指标 (TP, FP, FN, Precision, Recall, F1-Score, IoU)。

    参数:
        predicted_edge_map (np.ndarray): 算法输出的二值边缘图 (0 代表背景, 255 代表边缘)。
        ground_truth_map (np.ndarray): 真实的二值边缘图 (0 代表背景, 255 代表边缘)。

    返回:
        dict: 包含 TP, FP, FN, Precision, Recall, F1-Score, IoU 的字典。
              如果输入图像无效或维度不匹配，则返回包含错误信息的字典。
    """
    if predicted_edge_map is None or ground_truth_map is None:
        return {
            "TP": 0, "FP": 0, "FN": 0,
            "Precision": 0.0, "Recall": 0.0, "F1-Score": 0.0, "IoU": 0.0,
            "错误": "输入图像为空"
        }

    # 确保是二值图 (为了计算方便，内部转为 0 或 1)
    # 假设输入图像中 0 为背景，非零值为边缘
    pred_binary = (predicted_edge_map > 0).astype(np.uint8)
    gt_binary = (ground_truth_map > 0).astype(np.uint8)

    if pred_binary.shape != gt_binary.shape:
        return {
            "TP": 0, "FP": 0, "FN": 0,
            "Precision": 0.0, "Recall": 0.0, "F1-Score": 0.0, "IoU": 0.0,
            "错误": "预测图与真实图的维度不匹配。"
        }

    TP = np.sum((pred_binary == 1) & (gt_binary == 1))
    FP = np.sum((pred_binary == 1) & (gt_binary == 0))
    FN = np.sum((pred_binary == 0) & (gt_binary == 1))
    # TN = np.sum((pred_binary == 0) & (gt_binary == 0)) # TN 通常不直接用于 F1/IoU

    precision = TP / (TP + FP) if (TP + FP) > 0 else 0.0
    recall = TP / (TP + FN) if (TP + FN) > 0 else 0.0
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
    iou = TP / (TP + FP + FN) if (TP + FP + FN) > 0 else 0.0

    return {
        "TP": int(TP), "FP": int(FP), "FN": int(FN),
        "查准率 (Precision)": float(precision),
        "查全率 (Recall)": float(recall),
        "F1分数 (F1-Score)": float(f1_score),
        "交并比 (IoU)": float(iou)
    }


def calculate_ssim(predicted_edge_map, ground_truth_map):
    """
    计算两幅二值边缘图之间的结构相似性 (SSIM)。

    参数:
        predicted_edge_map (np.ndarray): 算法输出的二值边缘图 (0-255)。
        ground_truth_map (np.ndarray): 真实的二值边缘图 (0-255)。

    返回:
        float: SSIM 值。如果图像无效则返回 0.0。
    """
    if predicted_edge_map is None or ground_truth_map is None or \
            predicted_edge_map.shape != ground_truth_map.shape:
        return 0.0

    # SSIM 要求图像至少有 7x7 的尺寸（默认 win_size=7）
    # 对于非常小的图像，可能需要调整 win_size 或处理异常
    min_dim = min(predicted_edge_map.shape)
    win_size = min(7, min_dim if min_dim % 2 == 1 else min_dim - 1)  # 确保 win_size 是奇数且小于等于图像维度

    if win_size < 3:  # 如果图像太小，无法计算有意义的SSIM
        return 0.0  # 或者返回一个标记值

    try:
        # data_range 是像素值的范围
        # 对于二值图，虽然不是标准用法，但可以计算
        return ssim(ground_truth_map, predicted_edge_map, data_range=255, win_size=win_size)
    except ValueError as e:
        print(f"计算 SSIM 时出错: {e}。可能是图像尺寸过小或 win_size 设置问题。")
        return 0.0  # 或者其他错误指示值


def calculate_psnr(predicted_edge_map, ground_truth_map):
    """
    计算两幅二值边缘图之间的峰值信噪比 (PSNR)。

    参数:
        predicted_edge_map (np.ndarray): 算法输出的二值边缘图 (0-255)。
        ground_truth_map (np.ndarray): 真实的二值边缘图 (0-255)。

    返回:
        float: PSNR 值。如果图像无效则返回 0.0。如果图像完全相同，返回 inf。
    """
    if predicted_edge_map is None or ground_truth_map is None or \
            predicted_edge_map.shape != ground_truth_map.shape:
        return 0.0

    # data_range 是像素值的最大范围
    return psnr(ground_truth_map, predicted_edge_map, data_range=255)


def evaluate_all_metrics(predicted_edge_map, ground_truth_map):
    """
    计算所有定义的评价指标。

    参数:
        predicted_edge_map (np.ndarray): 算法输出的二值边缘图 (0 代表背景, 255 代表边缘)。
        ground_truth_map (np.ndarray): 真实的二值边缘图 (0 代表背景, 255 代表边缘)。

    返回:
        dict: 包含所有评价指标的字典。
    """
    all_metrics = {}

    pixel_metrics = calculate_pixel_metrics(predicted_edge_map, ground_truth_map)
    all_metrics.update(pixel_metrics)  # 合并字典

    # 如果基础像素指标计算出错，则不继续计算 SSIM/PSNR
    if "错误" in all_metrics and all_metrics["错误"]:
        all_metrics["SSIM"] = 0.0
        all_metrics["PSNR (dB)"] = 0.0
        return all_metrics

    # 确保传递给 SSIM 和 PSNR 的是 0-255 范围的图像
    pred_u8 = (predicted_edge_map > 0).astype(np.uint8) * 255
    gt_u8 = (ground_truth_map > 0).astype(np.uint8) * 255

    ssim_value = calculate_ssim(pred_u8, gt_u8)
    psnr_value = calculate_psnr(pred_u8, gt_u8)

    all_metrics["SSIM"] = float(ssim_value)
    all_metrics["PSNR (dB)"] = float(psnr_value)

    return all_metrics


if __name__ == '__main__':
    # 示例用法:
    # 创建一个简单的预测图和真实图
    pred = np.array([[0, 255, 0], [255, 255, 0], [0, 0, 255]], dtype=np.uint8)
    gt = np.array([[255, 255, 0], [0, 255, 0], [0, 0, 255]], dtype=np.uint8)

    # 为了测试 SSIM/PSNR，创建稍大一点的图像
    pred_large = np.zeros((10, 10), dtype=np.uint8)
    gt_large = np.zeros((10, 10), dtype=np.uint8)
    pred_large[2:5, 2:5] = 255
    gt_large[2:5, 2:5] = 255
    gt_large[5:7, 5:7] = 255  # 增加一些差异

    print("--- 小图像测试 ---")
    metrics_small = evaluate_all_metrics(pred, gt)
    for key, value in metrics_small.items():
        print(f"{key}: {value}")

    print("\n--- 稍大图像测试 ---")
    metrics_large = evaluate_all_metrics(pred_large, gt_large)
    for key, value in metrics_large.items():
        print(f"{key}: {value}")

    print("\n--- 完全相同的图像测试 ---")
    metrics_identical = evaluate_all_metrics(gt_large, gt_large.copy())  # 使用 .copy()
    for key, value in metrics_identical.items():
        print(f"{key}: {value}")  # PSNR 应该为 inf

    # 预期输出 (近似值, SSIM/PSNR 取决于 skimage 版本和具体实现细节):
    # --- 小图像测试 ---
    # TP: 3
    # FP: 1
    # FN: 1
    # 查准率 (Precision): 0.75
    # 查全率 (Recall): 0.75
    # F1分数 (F1-Score): 0.75
    # 交并比 (IoU): 0.6
    # SSIM: (一个0到1之间的值, 对于小二值图可能不太稳定)
    # PSNR (dB): (一个值, 比如 10-30 dB 范围)

    # --- 稍大图像测试 ---
    # (具体值取决于 pred_large 和 gt_large 的内容)

    # --- 完全相同的图像测试 ---
    # ...
    # SSIM: 1.0
    # PSNR (dB): inf