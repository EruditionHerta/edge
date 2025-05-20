# core/data_loader.py
import os
import cv2
import numpy as np
from PyQt5.QtGui import QImage, QPixmap


def get_dataset_image_ids(data_dir="dataset/data", bon_dir="dataset/bon"):
    """
    扫描 data 和 bon 目录，查找对应的 jpg 图像并返回它们的 ID。
    假设图像名称是纯数字 (例如, '1.jpg', '2.jpg')。
    """
    if not os.path.exists(data_dir) or not os.path.exists(bon_dir):
        print(f"错误: 数据集目录 '{data_dir}' 或 '{bon_dir}' 未找到。")
        return []

    data_files = {os.path.splitext(f)[0] for f in os.listdir(data_dir) if f.endswith('.jpg')}
    bon_files = {os.path.splitext(f)[0] for f in os.listdir(bon_dir) if f.endswith('.jpg')}

    # 查找在 data 和 bon 目录中都存在的共同图像 ID
    common_ids = sorted(list(data_files.intersection(bon_files)), key=lambda x: int(x) if x.isdigit() else x)

    if not common_ids:
        print("在 data 和 bon 目录中未找到匹配的图像 ID。")
    return common_ids


def load_image_and_ground_truth(image_id, data_dir="dataset/data", bon_dir="dataset/bon"):
    """
    加载原始图像及其对应的真实边缘图。

    参数:
        image_id (str): 图像的数字 ID (不含扩展名)。
        data_dir (str): 包含原始图像的目录路径。
        bon_dir (str): 包含真实边缘图的目录路径。

    返回:
        tuple: (original_image_cv, ground_truth_cv)
               original_image_cv 是 BGR 格式的 OpenCV 图像。
               ground_truth_cv 是灰度格式的 OpenCV 图像 (二值化, 0 或 255)。
               如果加载失败则返回 (None, None)。
    """
    img_path = os.path.join(data_dir, f"{image_id}.jpg")
    gt_path = os.path.join(bon_dir, f"{image_id}.jpg")

    if not os.path.exists(img_path):
        print(f"错误: 图像文件未找到: {img_path}")
        return None, None
    if not os.path.exists(gt_path):
        print(f"错误: 真实边缘图文件未找到: {gt_path}")
        return None, None

    original_image_cv = cv2.imread(img_path)  # 默认为 BGR
    ground_truth_raw_cv = cv2.imread(gt_path, cv2.IMREAD_GRAYSCALE)  # 以灰度模式加载

    if original_image_cv is None:
        print(f"错误: 无法读取图像: {img_path}")
        return None, None
    if ground_truth_raw_cv is None:
        print(f"错误: 无法读取真实边缘图像: {gt_path}")
        return None, None

    # 二值化真实边缘图: 确保它是 0 或 255
    # 通常做法: 边缘为白色 (255)，背景为黑色 (0)。
    # 如果您的真实边缘图格式不同，请调整阈值。
    _, ground_truth_cv_binary = cv2.threshold(ground_truth_raw_cv, 127, 255, cv2.THRESH_BINARY)

    return original_image_cv, ground_truth_cv_binary


def cv_to_qpixmap(cv_img, target_width=None, target_height=None):
    """将 OpenCV 图像 (BGR 或灰度) 转换为 QPixmap。"""
    if cv_img is None:
        return QPixmap()

    # 如果提供了目标尺寸，则调整图像大小
    if target_width and target_height:
        cv_img_resized = cv2.resize(cv_img, (target_width, target_height), interpolation=cv2.INTER_AREA)
    else:
        cv_img_resized = cv_img

    height, width = cv_img_resized.shape[:2]

    if len(cv_img_resized.shape) == 3:  # BGR 图像
        bytes_per_line = 3 * width
        q_img = QImage(cv_img_resized.data, width, height, bytes_per_line,
                       QImage.Format_RGB888).rgbSwapped()  # BGR 转 RGB
    elif len(cv_img_resized.shape) == 2:  # 灰度图像
        bytes_per_line = width
        q_img = QImage(cv_img_resized.data, width, height, bytes_per_line, QImage.Format_Grayscale8)
    else:
        return QPixmap()  # 不应发生此情况

    return QPixmap.fromImage(q_img)


if __name__ == '__main__':
    # 为测试创建虚拟数据集目录
    os.makedirs("dataset/data", exist_ok=True)
    os.makedirs("dataset/bon", exist_ok=True)

    # 为 data/1.jpg 创建一个虚拟的 2x2 黑色图像
    dummy_data_img = np.zeros((2, 2, 3), dtype=np.uint8)
    cv2.imwrite("dataset/data/1.jpg", dummy_data_img)
    # 为 bon/1.jpg 创建一个带有一个白色像素的虚拟 2x2 图像
    dummy_bon_img = np.zeros((2, 2), dtype=np.uint8)
    dummy_bon_img[0, 0] = 255
    cv2.imwrite("dataset/bon/1.jpg", dummy_bon_img)

    cv2.imwrite("dataset/data/2.jpg", dummy_data_img)  # bon/2.jpg 不存在

    ids = get_dataset_image_ids()
    print(f"可用的图像 ID: {ids}")  # 预期: ['1']

    if '1' in ids:
        orig, gt = load_image_and_ground_truth('1')
        if orig is not None and gt is not None:
            print("成功加载图像 '1'。")
            print(f"原始图像形状: {orig.shape}, 真实边缘图形状: {gt.shape}")
            print(f"真实边缘图 (应为二值):\n{gt}")
        else:
            print("加载图像 '1' 失败。")