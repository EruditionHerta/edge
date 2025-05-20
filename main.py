# main.py
import sys
from PyQt5.QtWidgets import QApplication
from gui.main_window import MainWindow # 从 gui 包导入 MainWindow 类

if __name__ == '__main__':
    app = QApplication(sys.argv) # 创建 QApplication 实例

    # 您可以在这里设置全局样式表 (可选)
    # app.setStyleSheet(QMainWindow { background-color: #f0f0f0; } /* 主窗口背景色 */
        # QGroupBox { font-weight: bold; margin-top: 10px; } /* 组框样式 */
        # QGroupBox::title { subcontrol-origin: margin; subcontrol-position: top left; padding: 0 5px; }
        # QPushButton { padding: 5px; background-color: #e0e0e0; border: 1px solid #c0c0c0; border-radius: 3px;}
        # QPushButton:hover { background-color: #d0d0d0; }
        # QPushButton:pressed { background-color: #c0c0c0; }
        # )

    main_win = MainWindow() # 创建主窗口实例
    main_win.show() # 显示主窗口
    sys.exit(app.exec_()) # 启动 Qt 应用的事件循环