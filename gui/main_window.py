# gui/main_window.py
import sys
import os
import pandas as pd  # 导入 pandas 用于 CSV 导出
from PyQt5.QtWidgets import (QMainWindow, QApplication, QWidget, QVBoxLayout, QHBoxLayout,
                             QListWidget, QComboBox, QPushButton, QLabel, QGroupBox, QGridLayout,
                             QFileDialog, QMessageBox, QTableWidget, QTableWidgetItem, QAbstractItemView,
                             QSplitter, QLineEdit, QFormLayout, QTabWidget, QTextEdit)
from PyQt5.QtCore import Qt, QThread, pyqtSignal
import cv2

# 从项目中导入模块
from core.data_loader import get_dataset_image_ids, load_image_and_ground_truth, cv_to_qpixmap
from core.algorithms_builtin import apply_canny, apply_sobel, apply_laplacian
from core.custom_algorithm_handler import run_custom_script
from core.evaluation import evaluate_all_metrics
from gui.matplotlib_widget import MatplotlibWidget


# --- 用于耗时操作的工作线程 (与之前相同) ---
class ProcessingThread(QThread):
    finished_signal = pyqtSignal(object, object, object, object, object)

    def __init__(self, original_img_cv, ground_truth_cv, algorithm_name, params):
        super().__init__()
        self.original_img_cv = original_img_cv
        self.ground_truth_cv = ground_truth_cv
        self.algorithm_name = algorithm_name
        self.params = params

    def run(self):
        binary_result_cv = None
        confidence_map_cv = None
        metrics_dict = None
        pr_recalls = None
        pr_precisions = None
        error_str = None

        try:
            if self.original_img_cv is None:
                error_str = "原始图像未加载。"
                self.finished_signal.emit(None, None, None, None, error_str)
                return

            output_from_algo = None
            if self.algorithm_name == "Canny":
                binary_result_cv, confidence_map_cv = apply_canny(self.original_img_cv,
                                                                  self.params.get('threshold1', 100),
                                                                  self.params.get('threshold2', 200),
                                                                  self.params.get('aperture_size', 3))
            elif self.algorithm_name == "Sobel":
                binary_result_cv, confidence_map_cv = apply_sobel(self.original_img_cv,
                                                                  self.params.get('ksize', 3),
                                                                  threshold_val=self.params.get('threshold_val', 50))
            elif self.algorithm_name == "Laplacian":
                binary_result_cv, confidence_map_cv = apply_laplacian(self.original_img_cv,
                                                                      self.params.get('ksize', 3),
                                                                      threshold_val=self.params.get('threshold_val',
                                                                                                    20))
            elif self.algorithm_name == "Custom" and 'script_path' in self.params:
                script_path = self.params['script_path']
                if not script_path or not os.path.exists(script_path):
                    error_str = f"自定义脚本路径无效或未设置: {script_path}"
                else:
                    binary_result_cv, confidence_map_cv, custom_error = run_custom_script(
                        script_path,
                        self.original_img_cv.copy()
                    )
                    if custom_error:
                        error_str = f"自定义脚本错误: {custom_error}"
            else:
                error_str = f"未知算法或缺少参数: {self.algorithm_name}"

            if binary_result_cv is not None and self.ground_truth_cv is not None:
                metrics_dict = evaluate_all_metrics(binary_result_cv, self.ground_truth_cv)
                if "错误" in metrics_dict and metrics_dict["错误"]:
                    current_error = metrics_dict["错误"]
                    error_str = (error_str + "; " if error_str else "") + f"二值图评价错误: {current_error}"
                    metrics_dict = None
            elif binary_result_cv is None and not error_str:
                error_str = (error_str + "; " if error_str else "") + "算法未生成二值输出图像。"

            if self.ground_truth_cv is None and binary_result_cv is not None:
                error_str = (error_str + "; " if error_str else "") + "真实边缘图未加载，无法评价标准指标。"
                metrics_dict = None

            if confidence_map_cv is not None and self.ground_truth_cv is not None:
                print("[处理线程] 1")
                pr_recalls = None
                pr_precisions = None
                if pr_recalls is None or pr_precisions is None:
                    print("[处理线程] 1")
                    pr_error_msg = "传统算子不建议生成PR曲线图"
                    error_str = (error_str + "; " if error_str else "") + pr_error_msg
                else:
                    print("1")

            elif confidence_map_cv is None and not error_str:
                pass
            elif self.ground_truth_cv is None and confidence_map_cv is not None:
                pr_error_msg = "真实边缘图未加载，无法生成PR曲线。"
                error_str = (error_str + "; " if error_str else "") + pr_error_msg

        except Exception as e:
            import traceback
            error_str = f"处理过程中发生严重错误: {str(e)}\n{traceback.format_exc()}"
            binary_result_cv = None
            metrics_dict = None
            pr_recalls = None
            pr_precisions = None

        self.finished_signal.emit(binary_result_cv, metrics_dict, pr_recalls, pr_precisions, error_str)


class MainWindow(QMainWindow):
    IMAGE_DISPLAY_WIDTH = 300
    IMAGE_DISPLAY_HEIGHT = 300

    def __init__(self):
        super().__init__()
        self.setWindowTitle("EdgeEval Pro - 图像边缘提取评价器")
        self.setGeometry(100, 100, 1250, 800)  # 稍微增加高度以容纳导出按钮

        self.current_original_cv = None
        self.current_ground_truth_cv = None
        self.current_result_cv = None  # 存储算法输出的二值边缘图 (OpenCV 格式)
        self.current_displayed_metrics = None  # 存储当前显示的指标字典
        self.custom_script_path = ""
        self.processing_thread = None

        self._init_ui()
        self.load_dataset_list()

    def _init_ui(self):
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        main_layout = QHBoxLayout(main_widget)

        left_panel = QWidget()
        left_layout = QVBoxLayout(left_panel)
        left_panel.setFixedWidth(380)

        # --- 数据集列表 (与之前相同) ---
        dataset_group = QGroupBox("数据集图像")
        dataset_layout = QVBoxLayout()
        self.image_list_widget = QListWidget()
        self.image_list_widget.currentTextChanged.connect(self.on_image_selected)
        dataset_layout.addWidget(self.image_list_widget)
        dataset_group.setLayout(dataset_layout)
        left_layout.addWidget(dataset_group)

        # --- 算法选择与参数 (与之前相同) ---
        algo_group = QGroupBox("算法选择与参数")
        algo_form_layout = QFormLayout()
        self.algo_combo = QComboBox()
        self.algo_combo.addItems(["Canny", "Sobel", "Laplacian", "Custom"])
        self.algo_combo.currentTextChanged.connect(self.on_algo_changed)
        algo_form_layout.addRow("选择算法:", self.algo_combo)
        # Canny, Sobel, Laplacian, Custom 参数组 (与之前相同)
        self.canny_params_group = QGroupBox("Canny 参数")  # ... (内容同前)
        canny_form = QFormLayout();
        self.canny_thresh1 = QLineEdit("100");
        self.canny_thresh2 = QLineEdit("200");
        self.canny_aperture = QComboBox();
        self.canny_aperture.addItems(["3", "5", "7"]);
        canny_form.addRow("阈值1:", self.canny_thresh1);
        canny_form.addRow("阈值2:", self.canny_thresh2);
        canny_form.addRow("孔径大小:", self.canny_aperture);
        self.canny_params_group.setLayout(canny_form)
        algo_form_layout.addRow(self.canny_params_group)
        self.sobel_params_group = QGroupBox("Sobel 参数");  # ... (内容同前)
        sobel_form = QFormLayout();
        self.sobel_ksize = QComboBox();
        self.sobel_ksize.addItems(["3", "5", "7"]);
        self.sobel_threshold = QLineEdit("50");
        sobel_form.addRow("核大小 (ksize):", self.sobel_ksize);
        sobel_form.addRow("二值化阈值:", self.sobel_threshold);
        self.sobel_params_group.setLayout(sobel_form)
        algo_form_layout.addRow(self.sobel_params_group);
        self.sobel_params_group.setVisible(False)
        self.laplacian_params_group = QGroupBox("Laplacian 参数");  # ... (内容同前)
        laplacian_form = QFormLayout();
        self.laplacian_ksize = QComboBox();
        self.laplacian_ksize.addItems(["1", "3", "5", "7"]);
        self.laplacian_threshold = QLineEdit("20");
        laplacian_form.addRow("核大小 (ksize):", self.laplacian_ksize);
        laplacian_form.addRow("二值化阈值:", self.laplacian_threshold);
        self.laplacian_params_group.setLayout(laplacian_form)
        algo_form_layout.addRow(self.laplacian_params_group);
        self.laplacian_params_group.setVisible(False)
        self.custom_algo_group = QGroupBox("自定义算法");  # ... (内容同前)
        custom_algo_layout_inner = QVBoxLayout();
        self.load_script_button = QPushButton("加载 Python 脚本 (.py)");
        self.load_script_button.clicked.connect(self.load_custom_script);
        self.custom_script_label = QLabel("未加载脚本。");
        self.custom_script_label.setWordWrap(True);
        custom_algo_layout_inner.addWidget(self.load_script_button);
        custom_algo_layout_inner.addWidget(self.custom_script_label);
        self.custom_algo_group.setLayout(custom_algo_layout_inner)
        algo_form_layout.addRow(self.custom_algo_group);
        self.custom_algo_group.setVisible(False)
        algo_group.setLayout(algo_form_layout)
        left_layout.addWidget(algo_group)

        # --- 运行按钮 (与之前相同) ---
        self.run_button = QPushButton("运行评价")
        self.run_button.setFixedHeight(40)
        self.run_button.setStyleSheet("QPushButton { font-size: 16px; background-color: lightgreen; }")
        self.run_button.clicked.connect(self.run_evaluation)
        left_layout.addWidget(self.run_button)

        # --- 新增：结果导出组 ---
        export_group = QGroupBox("结果导出")
        export_layout = QVBoxLayout()
        self.export_image_button = QPushButton("导出边缘图 (PNG)")
        self.export_image_button.clicked.connect(self.handle_export_edge_image)
        self.export_image_button.setEnabled(False)  # 初始禁用
        export_layout.addWidget(self.export_image_button)

        self.export_metrics_button = QPushButton("导出指标 (CSV)")
        self.export_metrics_button.clicked.connect(self.handle_export_metrics_csv)
        self.export_metrics_button.setEnabled(False)  # 初始禁用
        export_layout.addWidget(self.export_metrics_button)
        export_group.setLayout(export_layout)
        left_layout.addWidget(export_group)

        left_layout.addStretch()
        left_panel.setLayout(left_layout)

        # --- 右侧面板 (与之前相同，包含图像显示和指标/绘图) ---
        right_panel = QWidget()
        right_layout = QVBoxLayout(right_panel)
        image_tabs = QTabWidget()
        self.orig_img_label = QLabel("原始图像");
        self.orig_img_label.setAlignment(Qt.AlignCenter);
        self.orig_img_label.setFixedSize(self.IMAGE_DISPLAY_WIDTH, self.IMAGE_DISPLAY_HEIGHT);
        self.orig_img_label.setStyleSheet("border: 1px solid gray;")
        image_tabs.addTab(self.create_scrollable_label_wrapper(self.orig_img_label), "原始图像")
        self.gt_img_label = QLabel("真实边缘图");
        self.gt_img_label.setAlignment(Qt.AlignCenter);
        self.gt_img_label.setFixedSize(self.IMAGE_DISPLAY_WIDTH, self.IMAGE_DISPLAY_HEIGHT);
        self.gt_img_label.setStyleSheet("border: 1px solid gray;")
        image_tabs.addTab(self.create_scrollable_label_wrapper(self.gt_img_label), "真实边缘")
        self.result_img_label = QLabel("算法输出边缘");
        self.result_img_label.setAlignment(Qt.AlignCenter);
        self.result_img_label.setFixedSize(self.IMAGE_DISPLAY_WIDTH, self.IMAGE_DISPLAY_HEIGHT);
        self.result_img_label.setStyleSheet("border: 1px solid gray;")
        image_tabs.addTab(self.create_scrollable_label_wrapper(self.result_img_label), "算法结果")
        image_tabs.setFixedHeight(self.IMAGE_DISPLAY_HEIGHT + 50)
        right_layout.addWidget(image_tabs)
        results_splitter = QSplitter(Qt.Vertical)
        metrics_group = QGroupBox("评价指标");
        metrics_layout = QVBoxLayout();
        self.metrics_table = QTableWidget();
        self.metrics_table.setColumnCount(2);
        self.metrics_table.setHorizontalHeaderLabels(["指标名称", "值"]);
        self.metrics_table.setEditTriggers(QAbstractItemView.NoEditTriggers);
        self.metrics_table.setAlternatingRowColors(True);
        metrics_layout.addWidget(self.metrics_table);
        metrics_group.setLayout(metrics_layout)
        results_splitter.addWidget(metrics_group)
        plot_group = QGroupBox("指标可视化");
        plot_layout = QVBoxLayout();
        self.matplotlib_widget = MatplotlibWidget();
        plot_layout.addWidget(self.matplotlib_widget);
        plot_group.setLayout(plot_layout)
        results_splitter.addWidget(plot_group)
        results_splitter.setSizes([180, 280])
        right_layout.addWidget(results_splitter)
        self.status_label = QTextEdit("状态: 就绪。请选择图像和算法。");
        self.status_label.setReadOnly(True);
        self.status_label.setFixedHeight(80)
        right_layout.addWidget(self.status_label)
        right_panel.setLayout(right_layout)

        main_splitter = QSplitter(Qt.Horizontal)
        main_splitter.addWidget(left_panel)
        main_splitter.addWidget(right_panel)
        main_splitter.setSizes([380, 870])
        main_layout.addWidget(main_splitter)
        self.on_algo_changed(self.algo_combo.currentText())

    def create_scrollable_label_wrapper(self, label_widget):
        return label_widget

    def load_dataset_list(self):
        self.image_list_widget.clear()
        try:
            script_dir = os.path.dirname(os.path.abspath(__file__))
            base_dir = os.path.dirname(script_dir)
            data_dir = os.path.join(base_dir, "dataset/data")
            bon_dir = os.path.join(base_dir, "dataset/bon")
            image_ids = get_dataset_image_ids(data_dir, bon_dir)
            if image_ids:
                self.image_list_widget.addItems(image_ids)
                if self.image_list_widget.count() > 0:
                    self.image_list_widget.setCurrentRow(0)
            else:
                self.update_status("数据集中未找到图像，或 data/bon 文件夹内容不匹配。", error=True)
        except Exception as e:
            self.update_status(f"加载数据集列表时出错: {e}", error=True)
            QMessageBox.critical(self, "错误", f"无法加载数据集列表: {e}")

    def on_image_selected(self, image_id_str):
        # --- 禁用导出按钮 ---
        self.export_image_button.setEnabled(False)
        self.export_metrics_button.setEnabled(False)
        self.current_result_cv = None
        self.current_displayed_metrics = None
        # ---
        if not image_id_str:
            self.current_original_cv = None
            self.current_ground_truth_cv = None
            self.orig_img_label.clear();
            self.gt_img_label.clear()
            self.orig_img_label.setText("请选择一张图片");
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
            self.result_img_label.clear()
            self.result_img_label.setText("算法输出边缘")
            self.clear_metrics_and_plot()
            self.update_status(f"已加载: {image_id_str}.jpg")
        except Exception as e:
            self.update_status(f"加载图像 {image_id_str} 时出错: {e}", error=True)
            QMessageBox.warning(self, "加载错误", f"无法加载图像 {image_id_str}: {e}")

    def on_algo_changed(self, algo_name):
        self.canny_params_group.setVisible(algo_name == "Canny")
        self.sobel_params_group.setVisible(algo_name == "Sobel")
        self.laplacian_params_group.setVisible(algo_name == "Laplacian")
        self.custom_algo_group.setVisible(algo_name == "Custom")

    def load_custom_script(self):
        options = QFileDialog.Options()
        fileName, _ = QFileDialog.getOpenFileName(self, "加载自定义 Python 脚本", "",
                                                  "Python 文件 (*.py);;所有文件 (*)", options=options)
        if fileName:
            self.custom_script_path = fileName
            self.custom_script_label.setText(f"脚本: {os.path.basename(fileName)}")
            self.update_status(f"已加载自定义脚本: {fileName}")
        else:
            self.custom_script_path = ""
            self.custom_script_label.setText("未加载脚本。")

    def run_evaluation(self):
        # --- 禁用导出按钮 ---
        self.export_image_button.setEnabled(False)
        self.export_metrics_button.setEnabled(False)
        self.current_result_cv = None
        self.current_displayed_metrics = None
        # ---
        if self.processing_thread and self.processing_thread.isRunning():
            QMessageBox.warning(self, "正忙", "一个处理任务正在运行中，请稍候。")
            return
        if self.current_original_cv is None:
            QMessageBox.warning(self, "输入错误", "请先选择并加载一张原始图像。")
            return
        if self.current_ground_truth_cv is None:
            QMessageBox.warning(self, "输入错误", "真实边缘图未加载，无法进行评价。")
            return
        algo_name = self.algo_combo.currentText()
        params = {}
        try:
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
        self.run_button.setEnabled(False);
        self.run_button.setText("处理中...")
        self.update_status(f"正在运行 {algo_name}...")
        self.result_img_label.setText("处理中...")
        self.clear_metrics_and_plot()
        self.processing_thread = ProcessingThread(self.current_original_cv,
                                                  self.current_ground_truth_cv,
                                                  algo_name, params)
        self.processing_thread.finished_signal.connect(self.on_processing_finished)
        self.processing_thread.start()

    def on_processing_finished(self, result_img_cv, metrics_from_binary,
                               pr_recalls, pr_precisions, error_str):
        self.current_result_cv = result_img_cv
        self.current_displayed_metrics = metrics_from_binary  # 存储指标以供导出

        current_algo_name = self.algo_combo.currentText()

        if error_str:
            self.update_status(f"错误: {error_str}", error=True)
            self.result_img_label.setText(f"错误:\n{error_str[:150]}...")
            self.export_image_button.setEnabled(False)  # 出错则禁用导出图像
            self.export_metrics_button.setEnabled(False)  # 出错则禁用导出指标
        else:
            self.update_status("处理和评价完成。")
            if self.current_result_cv is not None:  # 仅当有结果图像时才启用导出图像按钮
                self.export_image_button.setEnabled(True)
            else:
                self.export_image_button.setEnabled(False)

            if self.current_displayed_metrics and not (
                    "错误" in self.current_displayed_metrics and self.current_displayed_metrics[
                "错误"]):  # 仅当有有效指标时启用导出指标
                self.export_metrics_button.setEnabled(True)
            else:
                self.export_metrics_button.setEnabled(False)

        if self.current_result_cv is not None:
            pixmap_res = cv_to_qpixmap(self.current_result_cv, self.IMAGE_DISPLAY_WIDTH, self.IMAGE_DISPLAY_HEIGHT)
            self.result_img_label.setPixmap(
                pixmap_res.scaled(self.result_img_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))
        elif not error_str:
            self.result_img_label.setText("无结果图像")

        plot_drawn = False
        if pr_recalls and pr_precisions and len(pr_recalls) > 1 and len(pr_precisions) > 1:
            try:
                self.matplotlib_widget.plot_pr_curve(pr_precisions, pr_recalls, current_algo_name)
                self.update_status(f"为 {current_algo_name} 绘制了PR曲线。")
                plot_drawn = True
            except Exception as e_plot:
                self.update_status(f"绘制PR曲线时出错: {e_plot}", error=True)
                self.matplotlib_widget.clear_plot()

        if not plot_drawn:
            if metrics_from_binary and not ("错误" in metrics_from_binary and metrics_from_binary["错误"]):
                self.matplotlib_widget.plot_metrics_bar(metrics_from_binary)
                self.update_status("绘制了指标条形图。")
            else:
                self.matplotlib_widget.clear_plot()
                if self.matplotlib_widget.ax: self.matplotlib_widget.ax.set_title("无可用绘图数据")
                self.matplotlib_widget.canvas.draw()

        if metrics_from_binary:
            self.display_metrics(metrics_from_binary)
        else:
            self.metrics_table.setRowCount(0)
            if not error_str:
                self.update_status("未能计算标准指标。", error=True)

        self.run_button.setEnabled(True)
        self.run_button.setText("运行评价")
        self.processing_thread = None

    def display_metrics(self, metrics_dict):
        self.metrics_table.setRowCount(0)
        if not metrics_dict or ("错误" in metrics_dict and metrics_dict["错误"]):
            self.metrics_table.setRowCount(1)
            self.metrics_table.setItem(0, 0, QTableWidgetItem("错误"))
            self.metrics_table.setItem(0, 1, QTableWidgetItem(metrics_dict.get("错误", "未知的评价错误")))
            return
        row = 0
        ordered_keys = [
            "TP", "FP", "FN",
            "查准率 (Precision)", "查全率 (Recall)", "F1分数 (F1-Score)",
            "交并比 (IoU)", "SSIM", "PSNR (dB)"
        ]
        for key in ordered_keys:
            if key in metrics_dict:
                value = metrics_dict[key]
                self.metrics_table.insertRow(row)
                self.metrics_table.setItem(row, 0, QTableWidgetItem(str(key)))
                if isinstance(value, float):
                    if value == float('inf'):
                        value_str = "inf"
                    else:
                        value_str = f"{value:.4f}"
                else:
                    value_str = str(value)
                self.metrics_table.setItem(row, 1, QTableWidgetItem(value_str))
                row += 1
        self.metrics_table.resizeColumnsToContents()

    def clear_metrics_and_plot(self, message_if_empty="选择图像并运行评价。"):
        self.metrics_table.setRowCount(0)
        self.matplotlib_widget.clear_plot()
        if self.matplotlib_widget.ax:
            self.matplotlib_widget.ax.set_title(message_if_empty)
            self.matplotlib_widget.canvas.draw()

    def update_status(self, message, error=False):
        if error:
            self.status_label.setTextColor(Qt.red)
        else:
            self.status_label.setTextColor(Qt.black)
        self.status_label.append(message)
        self.status_label.ensureCursorVisible()

        # --- 新增：导出功能槽函数 ---

    def handle_export_edge_image(self):
        """处理导出算法生成的边缘图像为PNG文件。"""
        if self.current_result_cv is None:
            QMessageBox.warning(self, "导出错误", "没有可导出的边缘图像。请先运行算法。")
            return

        current_img_item = self.image_list_widget.currentItem()
        if not current_img_item:
            QMessageBox.warning(self, "导出错误", "未选择图像。")
            return

        image_id = current_img_item.text()
        algo_name = self.algo_combo.currentText().replace(" ", "_").lower()
        default_filename = f"{image_id}_{algo_name}_edges.png"

        options = QFileDialog.Options()
        # options |= QFileDialog.DontUseNativeDialog
        filePath, _ = QFileDialog.getSaveFileName(self, "导出边缘图像为PNG", default_filename,
                                                  "PNG 文件 (*.png);;所有文件 (*)", options=options)
        if filePath:
            try:
                # OpenCV imwrite 需要 BGR 格式，但我们的边缘图通常是单通道灰度图
                # 如果是单通道，imwrite 会正确处理
                # 如果 current_result_cv 可能是彩色图（虽然这里不太可能），确保它是正确的
                if self.current_result_cv.ndim == 3 and self.current_result_cv.shape[2] == 3:
                    # 如果是BGR，则直接写入
                    success = cv2.imwrite(filePath, self.current_result_cv)
                elif self.current_result_cv.ndim == 2:
                    # 如果是单通道灰度图，直接写入
                    success = cv2.imwrite(filePath, self.current_result_cv)
                else:
                    self.update_status(f"错误: 无法识别的图像格式进行导出。", error=True)
                    QMessageBox.critical(self, "导出失败", "无法识别的图像格式。")
                    return

                if success:
                    self.update_status(f"边缘图像已成功导出到: {filePath}")
                    QMessageBox.information(self, "导出成功", f"边缘图像已保存到:\n{filePath}")
                else:
                    self.update_status(f"错误: 未能导出边缘图像到: {filePath}", error=True)
                    QMessageBox.critical(self, "导出失败", "未能保存图像。请检查文件路径和权限。")
            except Exception as e:
                self.update_status(f"导出边缘图像时发生错误: {e}", error=True)
                QMessageBox.critical(self, "导出异常", f"导出图像时发生错误:\n{e}")

    def handle_export_metrics_csv(self):
        """处理导出评价指标为CSV文件。"""
        if self.current_displayed_metrics is None or \
                ("错误" in self.current_displayed_metrics and self.current_displayed_metrics["错误"]):
            QMessageBox.warning(self, "导出错误", "没有有效的评价指标可导出。请先运行评价。")
            return

        current_img_item = self.image_list_widget.currentItem()
        if not current_img_item:
            QMessageBox.warning(self, "导出错误", "未选择图像。")
            return

        image_id = current_img_item.text()
        algo_name = self.algo_combo.currentText().replace(" ", "_").lower()
        default_filename = f"{image_id}_{algo_name}_metrics.csv"

        options = QFileDialog.Options()
        filePath, _ = QFileDialog.getSaveFileName(self, "导出评价指标为CSV", default_filename,
                                                  "CSV 文件 (*.csv);;所有文件 (*)", options=options)
        if filePath:
            try:
                # 将指标字典转换为适合CSV的格式 (两列：指标名称，值)
                metrics_list = []
                ordered_keys = [  # 与表格显示顺序一致
                    "TP", "FP", "FN",
                    "查准率 (Precision)", "查全率 (Recall)", "F1分数 (F1-Score)",
                    "交并比 (IoU)", "SSIM", "PSNR (dB)"
                ]
                for key in ordered_keys:
                    if key in self.current_displayed_metrics:
                        metrics_list.append({"指标名称": key, "值": self.current_displayed_metrics[key]})

                # 如果有其他不在ordered_keys中的指标也想导出，可以补充逻辑
                # for key, value in self.current_displayed_metrics.items():
                #     if key not in ordered_keys and key != "错误": # 排除错误信息
                #         metrics_list.append({"指标名称": key, "值": value})

                if not metrics_list:
                    QMessageBox.warning(self, "导出错误", "没有可导出的指标数据。")
                    return

                df = pd.DataFrame(metrics_list)
                df.to_csv(filePath, index=False, encoding='utf-8-sig')  # utf-8-sig 通常能更好处理Excel中的中文

                self.update_status(f"评价指标已成功导出到: {filePath}")
                QMessageBox.information(self, "导出成功", f"评价指标已保存到:\n{filePath}")
            except Exception as e:
                self.update_status(f"导出评价指标时发生错误: {e}", error=True)
                QMessageBox.critical(self, "导出异常", f"导出指标时发生错误:\n{e}")

    def closeEvent(self, event):
        if self.processing_thread and self.processing_thread.isRunning():
            reply = QMessageBox.question(self, '确认退出',
                                         "一个处理任务仍在运行中。您确定要退出吗？",
                                         QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
            if reply == QMessageBox.Yes:
                self.update_status("正在尝试终止处理线程...")
                self.processing_thread.terminate()
                self.processing_thread.wait(2000)
                if self.processing_thread.isRunning():
                    self.update_status("警告: 处理线程未能及时终止。", error=True)
                event.accept()
            else:
                event.ignore()
        else:
            event.accept()