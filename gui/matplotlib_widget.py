# gui/matplotlib_widget.py
from PyQt5.QtWidgets import QWidget, QVBoxLayout
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas  # Matplotlib 的 Qt5 后端
from matplotlib.figure import Figure  # Matplotlib 的 Figure 对象
import matplotlib.pyplot as plt  # 主要的绘图接口
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False


class MatplotlibWidget(QWidget):
    def __init__(self, parent=None):
        super(MatplotlibWidget, self).__init__(parent)
        # 创建一个 Figure 对象，这是所有绘图元素的顶层容器
        # figsize 设置图像大小（英寸），dpi 设置每英寸点数
        self.figure = Figure(figsize=(5, 4), dpi=100)
        # 创建一个 FigureCanvas 对象，它是 Figure 在 PyQt5 中的绘图区域
        self.canvas = FigureCanvas(self.figure)

        # 创建一个垂直布局，并将 canvas 添加到布局中
        layout = QVBoxLayout()
        layout.addWidget(self.canvas)
        self.setLayout(layout)  # 设置此 widget 的布局
        self.ax = None  # 用于存储 Axes 对象的占位符，Axes是实际绘图的区域

    def plot_metrics_bar(self, metrics_dict):
        """
        将查准率、查全率、F1分数绘制为条形图。
        metrics_dict 应包含 '查准率 (Precision)', '查全率 (Recall)', 'F1分数 (F1-Score)'。
        """
        if self.ax:
            self.figure.delaxes(self.ax)  # 清除之前的绘图区域

        self.ax = self.figure.add_subplot(111)  # 添加一个新的子图 (1行1列第1个)

        labels = ['查准率', '查全率', 'F1分数']
        # 从字典中获取指标值，如果键不存在则默认为 0.0
        values = [metrics_dict.get('查准率 (Precision)', 0.0),
                  metrics_dict.get('查全率 (Recall)', 0.0),
                  metrics_dict.get('F1分数 (F1-Score)', 0.0)]

        bars = self.ax.bar(labels, values, color=['skyblue', 'lightcoral', 'lightgreen'])  # 绘制条形图
        self.ax.set_ylim(0, 1.05)  # 设置 Y 轴范围 (指标通常在 0-1 之间)
        self.ax.set_ylabel('分数')  # 设置 Y 轴标签
        self.ax.set_title('评价指标条形图')  # 设置图表标题

        # 在每个条形图的顶部添加数值标签
        for bar in bars:
            yval = bar.get_height()  # 获取条形的高度
            # 在条形中心上方一点的位置显示文本
            self.ax.text(bar.get_x() + bar.get_width() / 2.0, yval + 0.01, f'{yval:.3f}', ha='center', va='bottom')

        self.canvas.draw()  # 刷新画布以显示绘图

    def plot_pr_curve(self, precision_values, recall_values, algorithm_name="算法"):
        """
        绘制 Precision-Recall (PR) 曲线。
        参数:
            precision_values (list or np.array): 查准率值列表。
            recall_values (list or np.array): 查全率值列表。
            algorithm_name (str): 算法名称，用于图例。
        """
        if self.ax:
            self.figure.delaxes(self.ax)  # 清除之前的绘图区域

        self.ax = self.figure.add_subplot(111)  # 添加新的子图
        self.ax.plot(recall_values, precision_values, marker='.', label=algorithm_name)  # 绘制 PR 曲线
        self.ax.set_xlabel('查全率 (Recall)')  # 设置 X 轴标签
        self.ax.set_ylabel('查准率 (Precision)')  # 设置 Y 轴标签
        self.ax.set_title('Precision-Recall 曲线')  # 设置图表标题
        self.ax.set_xlim([0.0, 1.0])  # 设置 X 轴范围
        self.ax.set_ylim([0.0, 1.05])  # 设置 Y 轴范围
        self.ax.grid(True)  # 显示网格
        self.ax.legend()  # 显示图例
        self.canvas.draw()  # 刷新画布

    def clear_plot(self):
        """清除当前的绘图区域。"""
        if self.ax:
            self.figure.delaxes(self.ax)  # 删除当前的 Axes 对象
            self.ax = None
        self.figure.clear()  # 清除整个 Figure 对象中的所有内容
        self.ax = self.figure.add_subplot(111)  # 添加一个默认的空子图
        self.ax.set_title("指标图")  # 设置默认标题
        self.canvas.draw()  # 刷新画布


if __name__ == '__main__':
    from PyQt5.QtWidgets import QApplication
    import sys
    import numpy as np  # 导入 numpy 用于 PR 曲线示例

    app = QApplication(sys.argv)  # 创建 PyQt5 应用
    main_widget = MatplotlibWidget()  # 创建 MatplotlibWidget 实例

    # 测试条形图
    sample_metrics = {'查准率 (Precision)': 0.85, '查全率 (Recall)': 0.78, 'F1分数 (F1-Score)': 0.81}
    main_widget.plot_metrics_bar(sample_metrics)

    # 测试 PR 曲线 (取消注释以测试)
    # main_widget.clear_plot() # 先清除之前的条形图
    # sample_recall = np.linspace(0, 1, 10) # 生成一些示例 recall 值
    # sample_precision = 1 - sample_recall**2 # 生成一些示例 precision 值 (通常 PR 曲线是下降的)
    # main_widget.plot_pr_curve(sample_precision, sample_recall, "测试 PR 曲线")

    main_widget.show()  # 显示 widget
    sys.exit(app.exec_())  # 运行应用事件循环