# gui/main_window.py
import sys
import os
from PyQt5.QtWidgets import (QMainWindow, QApplication, QWidget, QVBoxLayout, QHBoxLayout,
                             QListWidget, QComboBox, QPushButton, QLabel, QGroupBox, QGridLayout,
                             QFileDialog, QMessageBox, QTableWidget, QTableWidgetItem, QAbstractItemView,
                             QSplitter, QLineEdit, QFormLayout, QTabWidget, QTextEdit)
from PyQt5.QtCore import Qt, QThread, pyqtSignal
from PyQt5.QtGui import QPixmap, QFont  # QFont 未在此版本中使用，可移除

from core.data_loader import get_dataset_image_ids, load_image_and_ground_truth, cv_to_qpixmap
from core.algorithms_builtin import apply_canny, apply_sobel, apply_laplacian
from core.custom_algorithm_handler import run_custom_script
from core.evaluation import evaluate_all_metrics  # 更改为导入新的总评价函数
from gui.matplotlib_widget import MatplotlibWidget


# --- 用于耗时操作的工作线程 ---
class ProcessingThread(QThread):
    # 定义信号，参数为 (结果图像CV格式, 指标字典, 错误信息字符串)
    finished_signal = pyqtSignal(object, object, object)

    def __init__(self, original_img_cv, ground_truth_cv, algorithm_name, params):
        super().__init__()
        self.original_img_cv = original_img_cv
        self.ground_truth_cv = ground_truth_cv
        self.algorithm_name = algorithm_name
        self.params = params  # 算法参数字典

    def run(self):
        """线程执行的函数，进行图像处理和评价"""
        result_img_cv = None
        metrics = None
        error_str = None

        try:
            if self.original_img_cv is None:
                error_str = "原始图像未加载。"
                self.finished_signal.emit(None, None, error_str)
                return

            # 根据算法名称选择并执行算法
            if self.algorithm_name == "Canny":
                result_img_cv = apply_canny(self.original_img_cv,
                                            self.params.get('threshold1', 100),
                                            self.params.get('threshold2', 200),
                                            self.params.get('aperture_size', 3))
            elif self.algorithm_name == "Sobel":
                result_img_cv = apply_sobel(self.original_img_cv,
                                            self.params.get('ksize', 3),
                                            threshold_val=self.params.get('threshold_val', 50))
            elif self.algorithm_name == "Laplacian":
                result_img_cv = apply_laplacian(self.original_img_cv,
                                                self.params.get('ksize', 3),
                                                threshold_val=self.params.get('threshold_val', 20))
            elif self.algorithm_name == "Custom" and 'script_path' in self.params:
                script_path = self.params['script_path']
                if not script_path or not os.path.exists(script_path):
                    error_str = f"自定义脚本路径无效或未设置: {script_path}"
                else:
                    # 调用自定义脚本处理器
                    result_img_cv, custom_error = run_custom_script(script_path, self.original_img_cv.copy())
                    if custom_error:
                        error_str = f"自定义脚本错误: {custom_error}"
            else:
                error_str = f"未知算法或缺少参数: {self.algorithm_name}"

            # 如果算法成功生成结果图像，并且有真实边缘图，则进行评价
            if result_img_cv is not None and self.ground_truth_cv is not None:
                metrics = evaluate_all_metrics(result_img_cv, self.ground_truth_cv)  # 使用新的评价函数
                if "错误" in metrics and metrics["错误"]:  # 检查评价过程中是否有错误
                    error_str = (error_str + "; " if error_str else "") + metrics["错误"]
                    metrics = None  # 如果评价出错，则指标无效
            elif result_img_cv is None and not error_str:  # 算法未生成图像，但之前没有错误报告
                error_str = "算法未生成输出图像。"

            # 如果真实边缘图为空，则无法评价
            if self.ground_truth_cv is None and result_img_cv is not None:
                error_str = (error_str + "; " if error_str else "") + "真实边缘图未加载，无法进行评价。"
                metrics = None  # 确保指标为空


        except Exception as e:
            import traceback
            error_str = f"处理过程中发生错误: {str(e)}\n{traceback.format_exc()}"
            result_img_cv = None
            metrics = None

        self.finished_signal.emit(result_img_cv, metrics, error_str)  # 发送处理完成信号


class MainWindow(QMainWindow):
    IMAGE_DISPLAY_WIDTH = 300  # 图像显示区域的宽度
    IMAGE_DISPLAY_HEIGHT = 300  # 图像显示区域的高度

    def __init__(self):
        super().__init__()
        self.setWindowTitle("EdgeEval Pro - 图像边缘提取评价器")
        self.setGeometry(100, 100, 1250, 750)  # 窗口初始位置和大小

        self.current_original_cv = None  # 当前加载的 OpenCV 格式原始图像
        self.current_ground_truth_cv = None  # 当前加载的 OpenCV 格式真实边缘图
        self.current_result_cv = None  # 当前算法生成的 OpenCV 格式结果图像
        self.custom_script_path = ""  # 自定义算法脚本的路径
        self.processing_thread = None  # 用于处理的子线程

        self._init_ui()  # 初始化用户界面
        self.load_dataset_list()  # 加载数据集图像列表

    def _init_ui(self):
        """初始化用户界面元素。"""
        main_widget = QWidget()  # 主窗口的中心部件
        self.setCentralWidget(main_widget)
        main_layout = QHBoxLayout(main_widget)  # 主布局为水平布局

        # --- 左侧面板 (控制区域) ---
        left_panel = QWidget()
        left_layout = QVBoxLayout(left_panel)  # 左侧面板使用垂直布局
        left_panel.setFixedWidth(380)  # 设置左侧面板固定宽度

        # 数据集图像列表区域
        dataset_group = QGroupBox("数据集图像")
        dataset_layout = QVBoxLayout()
        self.image_list_widget = QListWidget()  # 用于显示图像 ID 的列表
        self.image_list_widget.currentTextChanged.connect(self.on_image_selected)  # 列表项改变时触发
        dataset_layout.addWidget(self.image_list_widget)
        dataset_group.setLayout(dataset_layout)
        left_layout.addWidget(dataset_group)

        # 算法选择区域
        algo_group = QGroupBox("算法选择与参数")
        algo_form_layout = QFormLayout()  # 使用表单布局方便标签和控件对齐

        self.algo_combo = QComboBox()  # 下拉框选择算法
        self.algo_combo.addItems(["Canny", "Sobel", "Laplacian", "Custom"])
        self.algo_combo.currentTextChanged.connect(self.on_algo_changed)  # 算法改变时触发
        algo_form_layout.addRow("选择算法:", self.algo_combo)

        # Canny 算子参数组
        self.canny_params_group = QGroupBox("Canny 参数")
        canny_form = QFormLayout()
        self.canny_thresh1 = QLineEdit("100")  # Canny 第一个阈值
        self.canny_thresh2 = QLineEdit("200")  # Canny 第二个阈值
        self.canny_aperture = QComboBox()  # Canny 孔径大小
        self.canny_aperture.addItems(["3", "5", "7"])
        canny_form.addRow("阈值1:", self.canny_thresh1)
        canny_form.addRow("阈值2:", self.canny_thresh2)
        canny_form.addRow("孔径大小:", self.canny_aperture)
        self.canny_params_group.setLayout(canny_form)
        algo_form_layout.addRow(self.canny_params_group)

        # Sobel 算子参数组
        self.sobel_params_group = QGroupBox("Sobel 参数")
        sobel_form = QFormLayout()
        self.sobel_ksize = QComboBox()  # Sobel 核大小
        self.sobel_ksize.addItems(["3", "5", "7"])  # ksize 必须是奇数
        self.sobel_threshold = QLineEdit("50")  # Sobel 二值化阈值
        sobel_form.addRow("核大小 (ksize):", self.sobel_ksize)
        sobel_form.addRow("二值化阈值:", self.sobel_threshold)
        self.sobel_params_group.setLayout(sobel_form)
        algo_form_layout.addRow(self.sobel_params_group)
        self.sobel_params_group.setVisible(False)  # 初始隐藏

        # Laplacian 算子参数组
        self.laplacian_params_group = QGroupBox("Laplacian 参数")
        laplacian_form = QFormLayout()
        self.laplacian_ksize = QComboBox()  # Laplacian 核大小
        self.laplacian_ksize.addItems(["1", "3", "5", "7"])
        self.laplacian_threshold = QLineEdit("20")  # Laplacian 二值化阈值
        laplacian_form.addRow("核大小 (ksize):", self.laplacian_ksize)
        laplacian_form.addRow("二值化阈值:", self.laplacian_threshold)
        self.laplacian_params_group.setLayout(laplacian_form)
        algo_form_layout.addRow(self.laplacian_params_group)
        self.laplacian_params_group.setVisible(False)  # 初始隐藏

        # 自定义算法组
        self.custom_algo_group = QGroupBox("自定义算法")
        custom_algo_layout_inner = QVBoxLayout()  # 这里用 QVBoxLayout 更合适
        self.load_script_button = QPushButton("加载 Python 脚本 (.py)")
        self.load_script_button.clicked.connect(self.load_custom_script)
        self.custom_script_label = QLabel("未加载脚本。")  # 显示已加载脚本的名称
        self.custom_script_label.setWordWrap(True)  # 自动换行
        custom_algo_layout_inner.addWidget(self.load_script_button)
        custom_algo_layout_inner.addWidget(self.custom_script_label)
        self.custom_algo_group.setLayout(custom_algo_layout_inner)
        algo_form_layout.addRow(self.custom_algo_group)  # 添加到主算法表单布局
        self.custom_algo_group.setVisible(False)  # 初始隐藏

        algo_group.setLayout(algo_form_layout)  # 设置算法选择区域的布局
        left_layout.addWidget(algo_group)

        self.run_button = QPushButton("运行评价")  # “运行/评价”按钮
        self.run_button.setFixedHeight(40)
        self.run_button.setStyleSheet("QPushButton { font-size: 16px; background-color: lightgreen; }")
        self.run_button.clicked.connect(self.run_evaluation)
        left_layout.addWidget(self.run_button)

        left_layout.addStretch()  # 添加伸缩因子，将控件推到顶部
        left_panel.setLayout(left_layout)

        # --- 右侧面板 (显示区域) ---
        right_panel = QWidget()
        right_layout = QVBoxLayout(right_panel)  # 右侧面板使用垂直布局

        # 图像显示标签页
        image_tabs = QTabWidget()

        self.orig_img_label = QLabel("原始图像")  # 显示原始图像的标签
        self.orig_img_label.setAlignment(Qt.AlignCenter)
        self.orig_img_label.setFixedSize(self.IMAGE_DISPLAY_WIDTH, self.IMAGE_DISPLAY_HEIGHT)
        self.orig_img_label.setStyleSheet("border: 1px solid gray;")
        image_tabs.addTab(self.create_scrollable_label_wrapper(self.orig_img_label), "原始图像")

        self.gt_img_label = QLabel("真实边缘图")  # 显示真实边缘图的标签
        self.gt_img_label.setAlignment(Qt.AlignCenter)
        self.gt_img_label.setFixedSize(self.IMAGE_DISPLAY_WIDTH, self.IMAGE_DISPLAY_HEIGHT)
        self.gt_img_label.setStyleSheet("border: 1px solid gray;")
        image_tabs.addTab(self.create_scrollable_label_wrapper(self.gt_img_label), "真实边缘")

        self.result_img_label = QLabel("算法输出边缘")  # 显示算法结果图像的标签
        self.result_img_label.setAlignment(Qt.AlignCenter)
        self.result_img_label.setFixedSize(self.IMAGE_DISPLAY_WIDTH, self.IMAGE_DISPLAY_HEIGHT)
        self.result_img_label.setStyleSheet("border: 1px solid gray;")
        image_tabs.addTab(self.create_scrollable_label_wrapper(self.result_img_label), "算法结果")

        image_tabs.setFixedHeight(self.IMAGE_DISPLAY_HEIGHT + 50)  # 为标签页栏留出一些额外高度
        right_layout.addWidget(image_tabs)

        # 指标和绘图显示区域 (使用垂直分割器)
        results_splitter = QSplitter(Qt.Vertical)

        # 评价指标表格
        metrics_group = QGroupBox("评价指标")
        metrics_layout = QVBoxLayout()
        self.metrics_table = QTableWidget()  # 用于显示指标的表格
        self.metrics_table.setColumnCount(2)
        self.metrics_table.setHorizontalHeaderLabels(["指标名称", "值"])
        self.metrics_table.setEditTriggers(QAbstractItemView.NoEditTriggers)  # 设置表格为只读
        self.metrics_table.setAlternatingRowColors(True)  # 启用交替行颜色
        metrics_layout.addWidget(self.metrics_table)
        metrics_group.setLayout(metrics_layout)
        results_splitter.addWidget(metrics_group)

        # Matplotlib 绘图区域
        plot_group = QGroupBox("指标可视化")
        plot_layout = QVBoxLayout()
        self.matplotlib_widget = MatplotlibWidget()  # 自定义的 Matplotlib 控件
        plot_layout.addWidget(self.matplotlib_widget)
        plot_group.setLayout(plot_layout)
        results_splitter.addWidget(plot_group)

        results_splitter.setSizes([180, 280])  # 设置表格和绘图区域的初始大小比例
        right_layout.addWidget(results_splitter)

        # 状态栏/日志区域
        self.status_label = QTextEdit("状态: 就绪。请选择图像和算法。")  # 显示状态信息的文本编辑框
        self.status_label.setReadOnly(True)  # 只读
        self.status_label.setFixedHeight(80)  # 固定高度
        right_layout.addWidget(self.status_label)

        right_panel.setLayout(right_layout)

        # 主分割器 (左右面板)
        main_splitter = QSplitter(Qt.Horizontal)
        main_splitter.addWidget(left_panel)
        main_splitter.addWidget(right_panel)
        main_splitter.setSizes([380, 870])  # 设置左右面板的初始大小比例

        main_layout.addWidget(main_splitter)
        self.on_algo_changed(self.algo_combo.currentText())  # 根据当前选择的算法更新参数组的可见性

    def create_scrollable_label_wrapper(self, label_widget):
        """为 QLabel 创建一个可滚动的包装器，以防图像过大 (虽然这里 QLabel 是固定大小的)。"""
        # from PyQt5.QtWidgets import QScrollArea # 如果需要滚动，取消注释
        # scroll_area = QScrollArea()
        # scroll_area.setWidgetResizable(True) # 允许内部 widget 调整大小
        # scroll_area.setWidget(label_widget)
        # return scroll_area
        return label_widget  # 由于 QLabel 已固定大小，直接返回

    def load_dataset_list(self):
        """加载数据集中的图像 ID 列表。"""
        self.image_list_widget.clear()  # 清空旧列表
        try:
            # 假设 'dataset/data' 和 'dataset/bon' 位于 main.py 同级或已知相对路径
            # 为了更稳健，您可能希望这些路径可配置或使用绝对路径
            script_dir = os.path.dirname(os.path.abspath(__file__))  # gui 目录
            base_dir = os.path.dirname(script_dir)  # EdgeEvalPro 项目根目录
            data_dir = os.path.join(base_dir, "dataset/data")
            bon_dir = os.path.join(base_dir, "dataset/bon")

            image_ids = get_dataset_image_ids(data_dir, bon_dir)
            if image_ids:
                self.image_list_widget.addItems(image_ids)
                if self.image_list_widget.count() > 0:
                    self.image_list_widget.setCurrentRow(0)  # 默认选中第一个
            else:
                self.update_status("数据集中未找到图像，或 data/bon 文件夹内容不匹配。", error=True)
        except Exception as e:
            self.update_status(f"加载数据集列表时出错: {e}", error=True)
            QMessageBox.critical(self, "错误", f"无法加载数据集列表: {e}")

    def on_image_selected(self, image_id_str):
        """当用户在列表中选择一个图像 ID 时调用。"""
        if not image_id_str:  # 如果没有选择任何项 (例如列表为空)
            self.current_original_cv = None
            self.current_ground_truth_cv = None
            self.orig_img_label.clear()
            self.gt_img_label.clear()
            self.orig_img_label.setText("请选择一张图片")
            self.gt_img_label.setText("请选择一张图片")
            return

        try:
            script_dir = os.path.dirname(os.path.abspath(__file__))
            base_dir = os.path.dirname(script_dir)
            data_dir = os.path.join(base_dir, "dataset/data")
            bon_dir = os.path.join(base_dir, "dataset/bon")

            self.current_original_cv, self.current_ground_truth_cv = \
                load_image_and_ground_truth(image_id_str, data_dir, bon_dir)

            if self.current_original_cv is not None:
                # 将 OpenCV 图像转换为 QPixmap 并显示
                pixmap = cv_to_qpixmap(self.current_original_cv, self.IMAGE_DISPLAY_WIDTH, self.IMAGE_DISPLAY_HEIGHT)
                self.orig_img_label.setPixmap(
                    pixmap.scaled(self.orig_img_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))
            else:
                self.orig_img_label.setText("加载原始图像失败")
                self.update_status(f"加载图像 {image_id_str} 的原始图像失败。", error=True)

            if self.current_ground_truth_cv is not None:
                pixmap_gt = cv_to_qpixmap(self.current_ground_truth_cv, self.IMAGE_DISPLAY_WIDTH,
                                          self.IMAGE_DISPLAY_HEIGHT)
                self.gt_img_label.setPixmap(
                    pixmap_gt.scaled(self.gt_img_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))
            else:
                self.gt_img_label.setText("加载真实边缘图失败")
                self.update_status(f"加载图像 {image_id_str} 的真实边缘图失败。", error=True)

            self.result_img_label.clear()  # 清除上一次的结果图像
            self.result_img_label.setText("算法输出边缘")
            self.clear_metrics_and_plot()  # 清除上一次的指标和绘图
            self.update_status(f"已加载: {image_id_str}.jpg")

        except Exception as e:
            self.update_status(f"加载图像 {image_id_str} 时出错: {e}", error=True)
            QMessageBox.warning(self, "加载错误", f"无法加载图像 {image_id_str}: {e}")

    def on_algo_changed(self, algo_name):
        """当选择的算法改变时，更新参数组的可见性。"""
        self.canny_params_group.setVisible(algo_name == "Canny")
        self.sobel_params_group.setVisible(algo_name == "Sobel")
        self.laplacian_params_group.setVisible(algo_name == "Laplacian")
        self.custom_algo_group.setVisible(algo_name == "Custom")

    def load_custom_script(self):
        """打开文件对话框让用户选择自定义算法的 Python 脚本。"""
        options = QFileDialog.Options()
        # options |= QFileDialog.DontUseNativeDialog # 如果需要非原生对话框
        fileName, _ = QFileDialog.getOpenFileName(self, "加载自定义 Python 脚本", "",
                                                  "Python 文件 (*.py);;所有文件 (*)", options=options)
        if fileName:
            self.custom_script_path = fileName
            self.custom_script_label.setText(f"脚本: {os.path.basename(fileName)}")  # 显示脚本文件名
            self.update_status(f"已加载自定义脚本: {fileName}")
        else:  # 如果用户取消选择
            self.custom_script_path = ""
            self.custom_script_label.setText("未加载脚本。")

    def run_evaluation(self):
        """执行边缘检测和评价流程。"""
        if self.processing_thread and self.processing_thread.isRunning():
            QMessageBox.warning(self, "正忙", "一个处理任务正在运行中，请稍候。")
            return

        if self.current_original_cv is None:
            QMessageBox.warning(self, "输入错误", "请先选择并加载一张原始图像。")
            return

        # 真实边缘图对于评价是必需的
        if self.current_ground_truth_cv is None:
            QMessageBox.warning(self, "输入错误", "真实边缘图未加载，无法进行评价。")
            return

        algo_name = self.algo_combo.currentText()  # 获取当前选择的算法名称
        params = {}  # 用于存储算法参数的字典

        try:  # 尝试获取并转换参数，处理可能的 ValueError
            if algo_name == "Canny":
                params['threshold1'] = int(self.canny_thresh1.text())
                params['threshold2'] = int(self.canny_thresh2.text())
                params['aperture_size'] = int(self.canny_aperture.currentText())
            elif algo_name == "Sobel":
                params['ksize'] = int(self.sobel_ksize.currentText())
                params['threshold_val'] = int(self.sobel_threshold.text())
            elif algo_name == "Laplacian":
                params['ksize'] = int(self.laplacian_ksize.currentText())
                params['threshold_val'] = int(self.laplacian_threshold.text())
            elif algo_name == "Custom":
                if not self.custom_script_path or not os.path.exists(self.custom_script_path):
                    QMessageBox.critical(self, "错误", "自定义脚本路径无效或未设置。")
                    return
                params['script_path'] = self.custom_script_path
        except ValueError as ve:
            QMessageBox.critical(self, "参数错误", f"参数值无效: {ve}")
            return

        # 更新UI，禁用按钮，显示处理中状态
        self.run_button.setEnabled(False)
        self.run_button.setText("处理中...")
        self.update_status(f"正在运行 {algo_name}...")
        self.result_img_label.setText("处理中...")  # 结果图像标签显示处理中
        self.clear_metrics_and_plot()  # 清除旧指标

        # 创建并启动处理线程
        self.processing_thread = ProcessingThread(self.current_original_cv,
                                                  self.current_ground_truth_cv,
                                                  algo_name, params)
        self.processing_thread.finished_signal.connect(self.on_processing_finished)  # 连接线程完成信号
        self.processing_thread.start()  # 启动线程

    def on_processing_finished(self, result_img_cv, metrics, error_str):
        """当处理线程完成时调用。"""
        self.current_result_cv = result_img_cv  # 保存结果图像

        if error_str:  # 如果有错误信息
            self.update_status(f"错误: {error_str}", error=True)
            # 在结果图像标签处显示部分错误信息
            self.result_img_label.setText(f"错误:\n{error_str[:150]}...")
        else:
            self.update_status("处理和评价完成。")

        if self.current_result_cv is not None:  # 如果成功生成结果图像
            pixmap_res = cv_to_qpixmap(self.current_result_cv, self.IMAGE_DISPLAY_WIDTH, self.IMAGE_DISPLAY_HEIGHT)
            self.result_img_label.setPixmap(
                pixmap_res.scaled(self.result_img_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))
        elif not error_str:  # 没有错误但也没有结果图像
            self.result_img_label.setText("无结果图像")

        if metrics:  # 如果有评价指标
            self.display_metrics(metrics)
            # 示例：使用条形图显示 Precision, Recall, F1。PR曲线需要更多数据点。
            self.matplotlib_widget.plot_metrics_bar(metrics)
        else:  # 如果没有指标 (可能是因为错误或真实图未加载)
            self.clear_metrics_and_plot(message_if_empty="无指标可显示。")

        # 恢复UI状态
        self.run_button.setEnabled(True)
        self.run_button.setText("运行评价")
        self.processing_thread = None  # 清除线程引用

    def display_metrics(self, metrics_dict):
        """在表格中显示评价指标。"""
        self.metrics_table.setRowCount(0)  # 清除之前的表格内容
        if not metrics_dict or ("错误" in metrics_dict and metrics_dict["错误"]):  # 检查字典是否有效或包含错误
            self.metrics_table.setRowCount(1)
            self.metrics_table.setItem(0, 0, QTableWidgetItem("错误"))
            self.metrics_table.setItem(0, 1, QTableWidgetItem(metrics_dict.get("错误", "未知的评价错误")))
            return

        row = 0
        # 定义期望显示的指标顺序
        ordered_keys = [
            "TP", "FP", "FN",
            "查准率 (Precision)", "查全率 (Recall)", "F1分数 (F1-Score)",
            "交并比 (IoU)", "SSIM", "PSNR (dB)"
        ]
        for key in ordered_keys:
            if key in metrics_dict:
                value = metrics_dict[key]
                self.metrics_table.insertRow(row)
                self.metrics_table.setItem(row, 0, QTableWidgetItem(str(key)))  # 指标名称
                # 格式化数值显示
                if isinstance(value, float):
                    if value == float('inf'):  # 处理 PSNR 可能为无穷大的情况
                        value_str = "inf"
                    else:
                        value_str = f"{value:.4f}"
                else:  # 对于 TP, FP, FN 等整数值
                    value_str = str(value)
                self.metrics_table.setItem(row, 1, QTableWidgetItem(value_str))  # 指标值
                row += 1
        self.metrics_table.resizeColumnsToContents()  # 自动调整列宽

    def clear_metrics_and_plot(self, message_if_empty="选择图像并运行评价。"):
        """清除指标表格和Matplotlib绘图区域。"""
        self.metrics_table.setRowCount(0)
        self.matplotlib_widget.clear_plot()
        if self.matplotlib_widget.ax:  # 确保ax存在
            self.matplotlib_widget.ax.set_title(message_if_empty)
            self.matplotlib_widget.canvas.draw()

    def update_status(self, message, error=False):
        """更新状态栏/日志区域的文本。"""
        if error:
            self.status_label.setTextColor(Qt.red)  # 错误信息用红色
        else:
            self.status_label.setTextColor(Qt.black)  # 正常信息用黑色 (或您的默认颜色)
        self.status_label.append(message)  # 追加信息到日志
        self.status_label.ensureCursorVisible()  # 自动滚动到底部

    def closeEvent(self, event):
        """处理窗口关闭事件，确保子线程正确退出。"""
        if self.processing_thread and self.processing_thread.isRunning():
            reply = QMessageBox.question(self, '确认退出',
                                         "一个处理任务仍在运行中。您确定要退出吗？",
                                         QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
            if reply == QMessageBox.Yes:
                self.update_status("正在尝试终止处理线程...")
                self.processing_thread.terminate()  # 强制终止线程 (可能不安全)
                self.processing_thread.wait(2000)  # 等待线程结束，设置超时
                if self.processing_thread.isRunning():
                    self.update_status("警告: 处理线程未能及时终止。", error=True)
                event.accept()  # 接受关闭事件
            else:
                event.ignore()  # 忽略关闭事件，不关闭窗口
        else:
            event.accept()  # 没有运行中的线程，直接接受关闭