# core/algorithms_builtin.py
import cv2
import numpy as np


def apply_canny(image_cv, threshold1=100, threshold2=200, aperture_size=3, l2_gradient=False):
    """
    应用 Canny 边缘检测。
    参数:
        image_cv (np.ndarray): 输入的 BGR 图像。
        threshold1 (int): 滞后过程的第一个阈值。
        threshold2 (int): 滞后过程的第二个阈值。
        aperture_size (int): Sobel 算子的孔径大小 (3, 5, 或 7)。
        l2_gradient (bool): 如果为 True, 则使用 L2 范数计算梯度幅值。
    返回:
        np.ndarray: 二值边缘图 (0 或 255)。
    """
    if image_cv is None: return None
    gray = cv2.cvtColor(image_cv, cv2.COLOR_BGR2GRAY)  # 转换为灰度图
    edges = cv2.Canny(gray, threshold1, threshold2,
                      apertureSize=aperture_size,
                      L2gradient=l2_gradient)
    return edges


def apply_sobel(image_cv, ksize=3, scale=1, delta=0, ddepth=cv2.CV_16S, threshold_val=50):
    """
    应用 Sobel 边缘检测。
    参数:
        image_cv (np.ndarray): 输入的 BGR 图像。
        ksize (int): Sobel 核的大小; 必须是 1, 3, 5, 或 7。
        scale (float): 计算出的导数值的可选缩放因子。
        delta (float): 可选的增量值，在存储结果之前添加到结果中。
        ddepth: 输出图像深度; 参考 cv2.Sobel 文档。使用 CV_16S 避免溢出。
        threshold_val (int): 用于二值化 Sobel 幅值的阈值。
    返回:
        np.ndarray: 二值边缘图 (0 或 255)。
    """
    if image_cv is None: return None
    gray = cv2.cvtColor(image_cv, cv2.COLOR_BGR2GRAY)  # 转换为灰度图

    grad_x = cv2.Sobel(gray, ddepth, 1, 0, ksize=ksize, scale=scale, delta=delta, borderType=cv2.BORDER_DEFAULT)
    grad_y = cv2.Sobel(gray, ddepth, 0, 1, ksize=ksize, scale=scale, delta=delta, borderType=cv2.BORDER_DEFAULT)

    abs_grad_x = cv2.convertScaleAbs(grad_x)  # 计算 x 方向梯度的绝对值
    abs_grad_y = cv2.convertScaleAbs(grad_y)  # 计算 y 方向梯度的绝对值

    grad = cv2.addWeighted(abs_grad_x, 0.5, abs_grad_y, 0.5, 0)  # 合并梯度

    _, edges = cv2.threshold(grad, threshold_val, 255, cv2.THRESH_BINARY)  # 应用阈值得到二值边缘图
    return edges


def apply_laplacian(image_cv, ksize=3, scale=1, delta=0, ddepth=cv2.CV_16S, threshold_val=20):
    """
    应用 Laplacian 边缘检测。
    参数:
        image_cv (np.ndarray): 输入的 BGR 图像。
        ksize (int): 用于计算二阶导数滤波器的孔径大小。
        threshold_val (int): 用于二值化的阈值。
    返回:
        np.ndarray: 二值边缘图 (0 或 255)。
    """
    if image_cv is None: return None
    gray = cv2.cvtColor(image_cv, cv2.COLOR_BGR2GRAY)  # 转换为灰度图

    # 可选: 使用高斯滤波器去噪
    # gray_blurred = cv2.GaussianBlur(gray, (3, 3), 0)

    laplacian = cv2.Laplacian(gray, ddepth, ksize=ksize, scale=scale, delta=delta, borderType=cv2.BORDER_DEFAULT)
    abs_laplacian = cv2.convertScaleAbs(laplacian)  # 计算拉普拉斯梯度的绝对值

    _, edges = cv2.threshold(abs_laplacian, threshold_val, 255, cv2.THRESH_BINARY)  # 应用阈值得到二值边缘图
    return edges


if __name__ == '__main__':
    # 为测试创建一个虚拟图像
    test_img = np.zeros((100, 100, 3), dtype=np.uint8)
    cv2.line(test_img, (20, 20), (80, 80), (0, 255, 0), 2)  # 一条绿线

    canny_edges = apply_canny(test_img)
    sobel_edges = apply_sobel(test_img)
    laplacian_edges = apply_laplacian(test_img)

    if canny_edges is not None:
        print("Canny 算子应用成功。")
        # cv2.imshow("Canny", canny_edges) # 需要 GUI 环境来显示
    if sobel_edges is not None:
        print("Sobel 算子应用成功。")
        # cv2.imshow("Sobel", sobel_edges)
    if laplacian_edges is not None:
        print("Laplacian 算子应用成功。")
        # cv2.imshow("Laplacian", laplacian_edges)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()