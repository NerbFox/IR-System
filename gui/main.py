import os
import sys
import logging
from multiprocessing import freeze_support

os.environ["QT_QUICK_BACKEND"] = "software"
os.environ["QTWEBENGINE_CHROMIUM_FLAGS"] = "--disable-gpu"
os.environ["QTWEBENGINE_DISABLE_SANDBOX"] = "1"

if not any(flag in sys.argv for flag in ["--pyside2", "--pyside6", "--pyqt5", "--pyqt6"]):
    sys.argv.append("--pyside6")

qt_toolkit = None

if '--pyside2' in sys.argv:
    from PySide2.QtWidgets import QApplication, QMainWindow, QFileDialog
    from PySide2.QtCore import QTimer, Qt, QCoreApplication
    from PySide2.QtGui import QIcon
    from PySide2.QtUiTools import QUiLoader
    qt_toolkit = "pyside2"

elif '--pyside6' in sys.argv:
    from PySide6.QtWidgets import QApplication, QMainWindow, QFileDialog, QWidget, QLabel, QVBoxLayout
    from PySide6.QtCore import QTimer, Qt, QCoreApplication, QFile
    from PySide6.QtGui import QIcon, QPixmap
    from PySide6.QtUiTools import QUiLoader
    qt_toolkit = "pyside6"

elif '--pyqt5' in sys.argv:
    from PyQt5.QtWidgets import QApplication, QMainWindow, QFileDialog
    from PyQt5.QtCore import QTimer, Qt, QCoreApplication, QFile
    from PyQt5 import uic, QtWebEngineWidgets
    from PyQt5.QtGui import QIcon
    qt_toolkit = "pyqt5"

elif '--pyqt6' in sys.argv:
    from PyQt6.QtWidgets import QApplication, QMainWindow, QFileDialog, QWidget, QLabel, QVBoxLayout
    from PyQt6.QtCore import QTimer, Qt, QCoreApplication
    from PyQt6.QtGui import QIcon
    from PyQt6 import uic, QtWebEngineWidgets
    qt_toolkit = "pyqt6"

from qt_material import apply_stylesheet, QtStyleTools, density

if qt_toolkit and hasattr(Qt, 'AA_ShareOpenGLContexts'):
    try:
        QCoreApplication.setAttribute(Qt.AA_ShareOpenGLContexts)
    except:
        QCoreApplication.set_attribute(Qt.AA_ShareOpenGLContexts)

app = QApplication([])
freeze_support()
try:
    app.processEvents()
    app.setQuitOnLastWindowClosed(False)
    app.lastWindowClosed.connect(app.quit)
except:
    app.process_events()
    app.quit_on_last_window_closed = False
    app.lastWindowClosed.connect(app.quit)

extra = {
    'danger': '#dc3545',
    'warning': '#ffc107',
    'success': '#17a2b8',
    'font_family': 'Roboto',
    'density_scale': '0',
    'button_shape': 'default',
}

class RuntimeStylesheets(QMainWindow, QtStyleTools):
    def __init__(self):
        super().__init__()
        
        if '--pyside2' in sys.argv:
            self.main = QUiLoader().load('main_window.ui', self)

        elif '--pyside6' in sys.argv:
            self.main = QUiLoader().load('main_window.ui', self)

        elif '--pyqt5' in sys.argv:
            self.main = uic.loadUi('main_window.ui', self)

        elif '--pyqt6' in sys.argv:
            self.main = uic.loadUi('main_window.ui', self)

        else:
            logging.error('must include --pyside2, --pyside6 or --pyqt5 in args!')
            sys.exit()

        self.main.radioButton_idf.toggled.connect(self.toggle_tf_method_visibility)
        self.main.radioButton_tf.toggled.connect(self.toggle_tf_method_visibility)
        self.main.radioButton_tfidf.toggled.connect(self.toggle_tf_method_visibility)
        self.main.radioButton_tfidfxnorm.toggled.connect(self.toggle_tf_method_visibility)
        self.main.radioButton_single.toggled.connect(self.toggle_input_mode_buttons)
        self.main.radioButton_batch.toggled.connect(self.toggle_input_mode_buttons)

        self.toggle_input_mode_buttons()
        self.toggle_tf_method_visibility()
        
        self.inverted_window = None
        self.main.pushButton_2.clicked.connect(self.show_inverted_file_window)

        try:
            self.main.setWindowTitle(f'{self.main.windowTitle()}')
        except:
            self.main.window_title = f'{self.main.window_title}'

        self.custom_styles()
        self.add_menu_theme(self.main, self.main.menuStyles)
        self.show_dock_theme(self.main)

        logo = QIcon("img/logo.png")

        try:
            self.main.setWindowIcon(logo)
        except:
            self.main.window_icon = logo

        if hasattr(QFileDialog, 'getExistingDirectory'):
            self.main.pushButton_file.clicked.connect(
                lambda: QFileDialog.getOpenFileName(self.main)
            )
            self.main.pushButton_folder.clicked.connect(
                lambda: QFileDialog.getExistingDirectory(self.main)
            )
        else:
            self.main.pushButton_file.clicked.connect(
                lambda: QFileDialog.get_open_file_name(self.main)
            )
            self.main.pushButton_folder.clicked.connect(
                lambda: QFileDialog.get_existing_directory(self.main)
            )

    def custom_styles(self):
        try:
            if hasattr(self.main, "tableWidget_2"):
                for r in range(self.main.tableWidget_2.rowCount()):
                    self.main.tableWidget_2.setRowHeight(r, 36)

        except Exception as e:
            print("Gagal set row height:", e)
    
    def toggle_tf_method_visibility(self):
        is_idf = self.main.radioButton_idf.isChecked()
        self.main.groupBox_tf_method.setDisabled(is_idf)
    
    def toggle_input_mode_buttons(self):
        is_single = self.main.radioButton_single.isChecked()
        self.main.pushButton_file.setDisabled(is_single)
        self.main.pushButton_process.setDisabled(not is_single)
        
    def show_inverted_file_window(self):
        self.inverted_window = InvertedFileWindow()
        self.inverted_window.show()





class InvertedFileWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Inverted File Viewer")
        self.setAttribute(Qt.WA_DeleteOnClose, True)
        
        logo = QIcon("img/logo.png")
        self.setWindowIcon(logo)

        if '--pyside6' in sys.argv:
            loader = QUiLoader()
            ui_file = os.path.abspath("inverted_window.ui")
            loaded = loader.load(ui_file, None)
            if loaded is None:
                raise RuntimeError(f"Failed to load UI file: {ui_file}")
            self.setCentralWidget(loaded)

        elif '--pyqt5' in sys.argv or '--pyqt6' in sys.argv:
            ui = uic.loadUi('inverted_window.ui')
            self.setCentralWidget(ui)

        else:
            raise RuntimeError("Unsupported Qt version.")


T0 = 1000

if __name__ == "__main__":
    def take_screenshot():
        pixmap = frame.main.grab()
        pixmap.save(os.path.join('screenshots', f'{theme}.png'))
        print(f'Saving {theme}')

    if len(sys.argv) > 2:
        theme = sys.argv[2]
        try:
            QTimer.singleShot(T0, take_screenshot)
            QTimer.singleShot(T0 * 2, app.closeAllWindows)
        except:
            QTimer.single_shot(T0, take_screenshot)
            QTimer.single_shot(T0 * 2, app.closeAllWindows)
    else:
        theme = 'default'

    apply_stylesheet(
        app,
        theme + '.xml',
        invert_secondary=('light' in theme and 'dark' not in theme),
        extra=extra,
    )

    frame = RuntimeStylesheets()
    frame.show() 
    frame.main.showMaximized()

    if hasattr(app, 'exec'):
        app.exec()
    else:
        app.exec_()
