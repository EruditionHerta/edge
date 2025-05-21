# core/custom_algorithm_handler.py
import importlib.util
import os
import numpy as np
import torch
import cv2


def run_custom_script(script_path, image_cv, device_str=None):
    """
    加载并运行自定义 Python 脚本进行边缘检测。
    脚本必须包含一个函数:
    `process_image(image_numpy_array, device)`
    该函数返回一个二值边缘图 (NumPy 数组, 0 或 255)。

    参数:
        script_path (str): 自定义 Python 脚本的路径。
        image_cv (np.ndarray): 输入的 BGR OpenCV 图像。
        device_str (str, 可选): "cuda" 或 "cpu"。如果为 None，则自动检测。

    返回:
        np.ndarray: 处理后的二值边缘图，如果发生错误则为 None。
        str: 错误信息 (如果有)，否则为 None。
    """
    if not os.path.exists(script_path):
        return None, None, f"自定义脚本未找到: {script_path}"

    try:
        # 从文件路径加载模块规范
        spec = importlib.util.spec_from_file_location("custom_module", script_path)
        if spec is None or spec.loader is None:  # 检查 spec 和 loader 是否有效
            return None, f"无法为自定义脚本加载规范: {script_path}", None
        custom_module = importlib.util.module_from_spec(spec)  # 根据规范创建模块
        spec.loader.exec_module(custom_module)  # 执行模块代码

        if not hasattr(custom_module, "process_image"):
            return None, None, "自定义脚本必须包含 'process_image(image_numpy_array, device)' 函数。"

        # 确定使用的设备 (GPU 或 CPU)
        if device_str:
            device = torch.device(device_str)
        else:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        print(f"自定义脚本将使用设备: {device}")

        # 确保传递给脚本的图像是副本，以防脚本内部进行原地修改
        image_to_process = image_cv.copy()

        # 调用自定义脚本中的处理函数
        edge_map = custom_module.process_image(image_to_process, device)

        if not isinstance(edge_map, np.ndarray):
            return None, None, "自定义脚本的 process_image 函数未返回 NumPy 数组。"

        # 确保输出是二值的 (0 或 255) 2D 灰度图像
        if edge_map.ndim == 3 and edge_map.shape[2] == 3:  # 如果是 BGR 图像
            edge_map = cv2.cvtColor(edge_map, cv2.COLOR_BGR2GRAY)  # 转为灰度图
        elif edge_map.ndim == 3 and edge_map.shape[2] == 1:  # 如果是 (H, W, 1) 形状
            edge_map = edge_map.squeeze(axis=2)  # 移除单维度

        if edge_map.ndim != 2:
            return None, None, "自定义脚本的输出必须是 2D 灰度图像。"

        # 二值化处理: 确保输出是 0 或 255
        # 假设非零值为边缘。
        # 如果脚本已返回 0/255, 此操作无害。
        # 如果脚本返回概率图, 此操作会在 >0 处进行阈值处理。
        # 对于更复杂的阈值处理, 脚本应自行处理或返回概率图，由主应用提供阈值化选项。
        if np.max(edge_map) == 1 and np.min(edge_map) == 0:  # 如果脚本返回 0 或 1
            edge_map_binary = (edge_map * 255).astype(np.uint8)
        else:  # 假设已经是 0 或其他值 (例如直接是 255)
            # 使用阈值确保是标准的二值图 (0 或 255)
            # 这里假设大于1的任何值都应该是边缘(255)
            _, edge_map_binary = cv2.threshold(edge_map.astype(np.uint8), 1, 255, cv2.THRESH_BINARY)

        return edge_map_binary, None, None

    except Exception as e:
        import traceback  # 导入 traceback 模块
        tb_str = traceback.format_exc()  # 获取详细的 traceback 信息
        error_message = f"执行自定义脚本时出错: {str(e)}\nTraceback:\n{tb_str}"
        print(error_message)  # 在控制台打印详细错误，方便调试
        return None, None, f"执行自定义脚本时出错: {str(e)}"  # 返回给GUI的可以是简化错误信息


if __name__ == '__main__':
    # 为测试创建一个虚拟的自定义脚本
    dummy_script_content = """
import numpy as np
import torch # 脚本可以使用 torch
import cv2   # 脚本可以使用 cv2

def process_image(image_numpy_array, device):
    print(f"自定义脚本接收到图像，形状: {image_numpy_array.shape}，设备: {device}")
    # 示例: 转换为灰度图并返回阈值化图像
    if image_numpy_array.ndim == 3:
        gray = cv2.cvtColor(image_numpy_array, cv2.COLOR_BGR2GRAY)
    else:
        gray = image_numpy_array

    # 虚拟处理: 阈值化
    _, edges = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)

    # 如果使用 PyTorch 模型:
    # tensor_img = torch.from_numpy(image_numpy_array).permute(2,0,1).float().unsqueeze(0).to(device)
    # model = YourModel().to(device)
    # with torch.no_grad():
    #    output_tensor = model(tensor_img)
    # edges_np = output_tensor.squeeze().cpu().numpy()
    # 将 edges_np 转换为二值的 0/255 np.uint8 数组

    print("自定义脚本处理完成。")
    return edges
"""
    os.makedirs("custom_scripts", exist_ok=True)
    dummy_script_path = "custom_scripts/dummy_test_script.py"
    with open(dummy_script_path, "w", encoding='utf-8') as f:  # 指定编码
        f.write(dummy_script_content)

    # 创建一个虚拟图像
    test_img_cv = np.random.randint(0, 256, (50, 50, 3), dtype=np.uint8)

    result_map, error_msg = run_custom_script(dummy_script_path, test_img_cv)

    if error_msg:
        print(f"错误: {error_msg}")
    else:
        print("自定义脚本执行成功。")
        if result_map is not None:
            print(f"结果图像形状: {result_map.shape}, 类型: {result_map.dtype}, 最大值: {np.max(result_map)}")
            # cv2.imshow("Custom Script Output", result_map) # 需要 GUI 环境
            # cv2.waitKey(0)
            # cv2.destroyAllWindows()
        else:
            print("结果图像为 None，但没有明确的错误信息。")