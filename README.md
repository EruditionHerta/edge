EdgeEval Pro - 图像边缘提取与评价软件简介

EdgeEval Pro 是一款用于图像边缘提取算法的综合性评价软件。它提供了一个用户友好的图形界面，允许用户加载标准数据集，选择内置的经典边缘提取算法，或者集成自定义的（支持GPU加速的PyTorch/OpenCV）算法进行边缘提取。软件能够自动计算多种常用的评价指标，并对结果进行可视化展示，包括原始图像、真实边缘图、算法输出图、指标表格、指标对比条形图以及Precision-Recall (PR) 曲线。本项目旨在为研究人员和开发者提供一个便捷的平台，以评估和比较不同边缘提取算法的性能。主要功能数据集管理:自动加载指定目录 (dataset/data 和 dataset/bon) 下的图像及其对应的真实边缘图 (Ground Truth)。支持 .jpg 格式的图像，图像文件名需为纯数字编号以建立对应关系。内置边缘提取算法:Canny 算子 (可调参数：阈值1, 阈值2, 孔径大小)Sobel 算子 (可调参数：核大小, 二值化阈值)Laplacian 算子 (可调参数：核大小, 二值化阈值)自定义算法支持:用户可以加载外部 Python 脚本 (.py) 作为自定义边缘提取算法。自定义脚本需遵循特定接口规范 (详见下文)。支持利用已安装的 PyTorch (包括CUDA GPU加速) 和 OpenCV 库。提供了一个使用预训练 VGG16 模型提取特征进行边缘检测的示例脚本。全面的评价指标:基于像素匹配: 真阳性 (TP), 假阳性 (FP), 假阴性 (FN)标准分类指标: 查准率 (Precision), 查全率 (Recall), F1分数 (F1-Score)分割常用指标: 交并比 (IoU / Jaccard Index)图像质量/相似度指标: 结构相似性 (SSIM), 峰值信噪比 (PSNR)结果可视化:并排显示原始图像、真实边缘图和算法输出的边缘图。以表格形式清晰展示所有计算出的评价指标。绘制条形图直观比较 Precision, Recall, F1-Score。为支持输出置信度图的算法绘制 Precision-Recall (PR) 曲线。用户界面:基于 PyQt5 构建的图形用户界面，操作直观。实时状态更新和错误提示。项目结构EdgeEvalPro/
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
├── dataset/                    # 存放基准数据集 (需用户自行创建和填充)
│   ├── data/                   # 存放原始图像 (例如 1.jpg, 2.jpg, ...)
│   └── bon/                    # 存放对应的真实边缘图 (例如 1.jpg, 2.jpg, ...)
├── custom_scripts/             # 存放用户自定义算法脚本的示例目录
│   ├── my_pytorch_edge_detector.py # 一个简单的PyTorch自定义脚本示例
│   └── vgg16_edge_detector.py      # 使用VGG16提取边缘的自定义脚本示例
└── requirements.txt            # 项目依赖的Python包列表
└── README.md                   # 本文档
环境要求Python 3.8+主要的 Python 库 (版本号仅供参考，请根据实际情况调整 requirements.txt):PyQt5 (例如 5.15.x)opencv-python (例如 4.x.x)numpy (例如 1.2x.x 或 2.x.x)matplotlib (例如 3.x.x)torch (例如 1.13.x, 建议带CUDA版本以支持GPU)torchvision (例如 0.14.x)scikit-image (例如 0.19.x 或更高)Pillowpandas (可选，当前代码未使用，但常见于数据处理)您可以通过以下命令安装依赖：pip install -r requirements.txt
对于 PyTorch，建议访问其官网根据您的 CUDA 版本选择合适的安装命令。安装与运行克隆或下载项目:获取项目所有文件。准备数据集:在项目根目录下创建 dataset 文件夹。在 dataset 文件夹内创建 data 和 bon 两个子文件夹。将您的原始图像 (例如 1.jpg, 2.jpg, ...) 放入 dataset/data/ 文件夹。将对应的真实边缘图 (Ground Truth，同样命名为 1.jpg, 2.jpg, ...) 放入 dataset/bon/ 文件夹。真实边缘图应为二值图像（例如，黑色背景，白色边缘）。安装依赖:如上所述，使用 pip install -r requirements.txt 安装所需库。确保 PyTorch 安装正确并能被您的系统识别（特别是GPU版本）。运行程序:在项目根目录下，执行以下命令：python main.py
自定义算法接口规范如果您想集成自己的边缘提取算法，需要创建一个 Python 脚本 (.py 文件) 并放置在例如 custom_scripts/ 目录下。该脚本必须包含一个具有以下签名的函数：def process_image(image_numpy_array, device):
    """
    处理输入的图像并返回边缘提取结果。

    参数:
        image_numpy_array (np.ndarray): 输入的原始图像。
                                       这是一个 NumPy 数组，格式为 BGR (OpenCV默认格式)，
                                       形状为 (Height, Width, Channels)。
        device (torch.device):        PyTorch 设备对象 (例如 torch.device('cuda:0') 或
                                       torch.device('cpu'))，由主程序根据GPU可用性传入。
                                       您可以在此设备上加载和运行您的PyTorch模型。

    返回:
        tuple 或者 np.ndarray:
        1. 推荐返回一个元组: `(binary_edge_map, confidence_map_or_None)`
           - binary_edge_map (np.ndarray): 最终的二值边缘图。
                                           形状为 (Height, Width)，数据类型为 np.uint8，
                                           值为 0 (背景) 或 255 (边缘)。
           - confidence_map_or_None (np.ndarray or None): 算法输出的置信度图 (可选)。
                                           如果提供，应为单通道灰度图 (Height, Width)，
                                           数据类型为 np.float32，值建议归一化到 [0, 1] 范围，
                                           表示每个像素是边缘的强度或概率。
                                           如果算法不生成置信度图，则返回 None。
                                           此图用于生成 PR 曲线。
        2. 为了向后兼容或简化，也可以只返回 `binary_edge_map` (np.ndarray)。
           这种情况下，软件将无法为该算法生成 PR 曲线。
    """
    # 您的算法实现...
    # 例如:
    # processed_binary_map = ...
    # processed_confidence_map = ... (如果适用)
    # return processed_binary_map, processed_confidence_map
    pass
示例自定义脚本:custom_scripts/my_pytorch_edge_detector.py: 一个简单的 PyTorch 模型示例。custom_scripts/vgg16_edge_detector.py: 使用预训练 VGG16 模型提取特征进行边缘检测的示例，它会返回二值图和置信度图。注意事项首次运行VGG16: vgg16_edge_detector.py 示例脚本在首次运行时会自动从 torchvision 下载预训练的 VGG16 模型权重，这可能需要一些时间，具体取决于您的网络连接。GPU内存: 运行基于深度学习的自定义算法（尤其是像VGG16这样较大的模型）时，请确保您的 GPU 有足够的显存。错误处理: 软件包含基本的错误提示，但自定义脚本中的内部错误可能需要您在脚本内部进行更详细的调试。自定义脚本执行的错误信息会尝试在状态栏中显示。PR曲线: 只有当算法（内置或自定义）能够输出置信度图时，才能为其生成 PR 曲线。Canny 算子目前不直接输出用于PR曲线的置信度图。希望这个 README.md 对您有所帮助！