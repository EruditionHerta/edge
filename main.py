# main.py
import sys
from PyQt5.QtWidgets import QApplication
from gui.main_window import MainWindow # 从 gui 包导入 MainWindow 类

if __name__ == '__main__':
    app = QApplication(sys.argv) # 创建 QApplication 实例

    main_win = MainWindow() # 创建主窗口实例
    main_win.show() # 显示主窗口
    sys.exit(app.exec_()) # 启动 Qt 应用的事件循环