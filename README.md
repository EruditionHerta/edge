# **EdgeEvalMax \- 基于BSDS500数据集的图像边缘提取与评价软件**

## **简介**

EdgeEvalMax 是一款用于图像边缘提取算法的综合性评价软件。它提供了一个用户友好的图形界面，允许用户加载BSDS500标准数据集，选择内置的经典边缘提取算法，或者集成自定义的（支持GPU加速的PyTorch/OpenCV）算法进行边缘提取。软件能够自动计算多种常用的评价指标，并对结果进行可视化展示，包括原始图像、真实边缘图、算法输出图、指标表格以及指标对比条形图。

本项目旨在为研究人员和开发者提供一个便捷的平台，以评估和比较不同边缘提取算法的性能。

## **主要功能**

* **数据集管理**:
  * 自动加载指定目录 (dataset/data 和 dataset/bon) 下的图像及其对应的真实边缘图 (Ground Truth)。
  * 支持 .jpg 格式的图像，图像文件名需为纯数字编号以建立对应关系。
* **内置边缘提取算法**:
  * Canny 算子 (可调参数：阈值1, 阈值2, 孔径大小)
  * Sobel 算子 (可调参数：核大小, 二值化阈值)
  * Laplacian 算子 (可调参数：核大小, 二值化阈值)
* **自定义算法支持**:
  * 用户可以加载外部 Python 脚本 (.py) 作为自定义边缘提取算法。
  * 自定义脚本需遵循特定接口规范 (详见下文)。
  * 支持利用已安装的 PyTorch (包括CUDA GPU加速) 和 OpenCV 库。
  * 提供了一个使用预训练 VGG16 模型提取特征进行边缘检测的示例脚本。
* **全面的评价指标**:
  * **基于像素匹配**: 真阳性 (TP), 假阳性 (FP), 假阴性 (FN)
  * **标准分类指标**: 查准率 (Precision), 查全率 (Recall), F1分数 (F1-Score)
  * **分割常用指标**: 交并比 (IoU / Jaccard Index)
  * **图像质量/相似度指标**: 结构相似性 (SSIM), 峰值信噪比 (PSNR)
* **结果可视化**:
  * 并排显示原始图像、真实边缘图和算法输出的边缘图。
  * 以表格形式清晰展示所有计算出的评价指标。
  * 绘制条形图直观比较 Precision, Recall, F1-Score。
* **用户界面**:
  * 基于 PyQt5 构建的图形用户界面，操作直观。
  * 实时状态更新和错误提示。

## **项目结构**

EdgeEvalMax/  
├── main.py                     # 主程序入口  
├── gui/                        # GUI 相关模块  
│   ├── __init__.py  
│   ├── main_window.py          # 主窗口UI和核心交互逻辑  
│   └── matplotlib_widget.py    # Matplotlib嵌入PyQt5的自定义控件  
├── core/                       # 核心功能模块  
│   ├── __init__.py  
│   ├── data_loader.py          # 数据集加载与图像转换  
│   ├── algorithms_builtin.py   # 内置算法实现  
│   ├── custom_algorithm_handler.py # 自定义算法加载与执行逻辑  
│   └── evaluation.py           # 评价指标计算与PR曲线数据生成  
├── dataset/                    # 存放BDSD500基准数据集  
│   ├── data/                   # 存放原始图像 (例如 1.jpg, 2.jpg, ...)  
│   └── bon/                    # 存放对应的真实边缘图 (例如 1.jpg, 2.jpg, ...)  
├── custom_scripts/             # 存放用户自定义算法脚本的示例目录  
│   ├── my_pytorch_edge_detector.py # 一个简单的PyTorch自定义脚本示例  
│   └── vgg16_edge_detector.py      # 使用VGG16提取边缘的自定义脚本示例  
├── requirements.txt            # 项目依赖的Python包列表  
└── README.md                   # 本文档  


## **环境要求**

* Python 3.9
* 主要的 Python 库:
  * PyQt5 (5.15.11)
  * opencv-python (4.11.0.86)
  * numpy (1.26.4)
  * matplotlib (3.9.4)
  * torch (1.13.1+cu116)
  * torchvision (0.14.1+cu116)
  * scikit-image (0.24.0)
* GPU支持: CUDA 11.6

您可以通过以下命令安装依赖：
```bash
pip install -r requirements.txt
```

对于 PyTorch，您需要访问其官网，根据您的 CUDA 版本选择合适的安装命令。
```
https://pytorch.org/get-started/previous-versions/
```

## **安装与运行**

1. **克隆或下载项目**:
   获取项目所有文件。
2. **安装依赖**:
   如上所述，使用 pip install \-r requirements.txt 安装所需库。确保 PyTorch 安装正确并能被您的系统识别（特别是GPU版本）。
3. **运行程序**:
   在项目根目录下，执行以下命令：
   ```bash
   python main.py
   ```

## **自定义算法接口规范**

如果您想集成自己的边缘提取算法，需要创建一个 Python 脚本 (.py 文件) 并放置在 custom\_scripts/ 目录下。该脚本必须包含一个具有以下签名的函数：
```python
def process_image(image_numpy_array: np.ndarray, device: torch.device) -> Union[np.ndarray, Tuple[np.ndarray, Optional[np.ndarray]]]:
    """
    处理输入图像并返回边缘提取结果。
    
    参数:
        image_numpy_array (np.ndarray):  
            输入的原始图像，格式为 **BGR**（OpenCV默认格式），  
            形状为 `(Height, Width, Channels)`，数据类型为 `np.uint8`。
        
        device (torch.device):  
            PyTorch 设备对象（如 `torch.device('cuda:0')` 或 `torch.device('cpu')`），  
            由主程序根据GPU可用性自动传入，用于模型加载和计算。
    
    返回:
        - **推荐格式**：元组 `(binary_edge_map, confidence_map_or_None)`  
          - `binary_edge_map` (np.ndarray):  
            二值边缘图，形状为 `(Height, Width)`，数据类型为 `np.uint8`，  
            值为 `0`（背景）或 `255`（边缘）。  
          - `confidence_map_or_None` (np.ndarray或None):  
            置信度图（可选），单通道灰度图，形状为 `(Height, Width)`，  
            数据类型为 `np.float32`，值需归一化到 `[0, 1]` 范围，  
            表示像素为边缘的强度或概率。若算法不生成置信度图，返回 `None`。  
            **注**：提供此图可用于生成PR曲线，否则无法生成。
        
        - **简化格式**（向后兼容）：仅返回 `binary_edge_map` (np.ndarray)，  
          此模式下软件将跳过PR曲线生成。
    """
    # 算法实现逻辑
    # 示例：
    # 1. 加载模型（建议在函数外执行，避免重复加载）
    # 2. 图像预处理（如转为RGB、归一化等）
    # 3. 使用device执行推理
    # 4. 后处理生成二值图和置信度图
    pass
```
**示例自定义脚本**:

* custom\_scripts/my\_pytorch\_edge\_detector.py: 一个简单的 PyTorch 模型示例。
* custom\_scripts/vgg16\_edge\_detector.py: 使用预训练 VGG16 模型提取特征进行边缘检测的示例，它会返回二值图和置信度图。

## **注意事项**

* **首次运行VGG16**: vgg16\_edge\_detector.py 示例脚本在首次运行时会自动从 torchvision 下载预训练的 VGG16 模型权重，这可能需要一些时间，具体取决于您的网络连接。
* **GPU内存**: 运行基于深度学习的自定义算法（尤其是像VGG16这样较大的模型）时，请确保您的 GPU 有足够的显存。
* **错误处理**: 软件包含基本的错误提示，但自定义脚本中的内部错误可能需要您在脚本内部进行更详细的调试。自定义脚本执行的错误信息会尝试在状态栏中显示。

希望这个说明文档对您有所帮助！
