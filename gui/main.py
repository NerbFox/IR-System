import os
import sys
import logging
from multiprocessing import freeze_support
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils import OriginalProcess, ExpandProcess
os.environ["QT_QUICK_BACKEND"] = "software"
os.environ["QTWEBENGINE_CHROMIUM_FLAGS"] = "--disable-gpu"
os.environ["QTWEBENGINE_DISABLE_SANDBOX"] = "1"

if not any(flag in sys.argv for flag in ["--pyside2", "--pyside6", "--pyqt5", "--pyqt6"]):
    sys.argv.append("--pyside6")

qt_toolkit = None

if '--pyside2' in sys.argv:
    from PySide2.QtWidgets import QApplication, QMainWindow, QFileDialog, QTableWidgetItem
    from PySide2.QtCore import QTimer, Qt, QCoreApplication
    from PySide2.QtGui import QIcon
    from PySide2.QtUiTools import QUiLoader
    qt_toolkit = "pyside2"

elif '--pyside6' in sys.argv:
    from PySide6.QtWidgets import QApplication, QMainWindow, QFileDialog, QWidget, QLabel, QVBoxLayout, QTableWidgetItem
    from PySide6.QtCore import QTimer, Qt, QCoreApplication, QFile
    from PySide6.QtGui import QIcon, QPixmap
    from PySide6.QtUiTools import QUiLoader
    qt_toolkit = "pyside6"

elif '--pyqt5' in sys.argv:
    from PyQt5.QtWidgets import QApplication, QMainWindow, QFileDialog, QTableWidgetItem
    from PyQt5.QtCore import QTimer, Qt, QCoreApplication, QFile
    from PyQt5 import uic, QtWebEngineWidgets
    from PyQt5.QtGui import QIcon
    qt_toolkit = "pyqt5"

elif '--pyqt6' in sys.argv:
    from PyQt6.QtWidgets import QApplication, QMainWindow, QFileDialog, QWidget, QLabel, QVBoxLayout,QTableWidgetItem
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

original_process = OriginalProcess()
expand_process = ExpandProcess()
is_fullbert = True

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
            
        
        self.ready_to_process = False
        self.main.radioButton_idf.toggled.connect(self.toggle_tf_method_visibility)
        self.main.radioButton_tf.toggled.connect(self.toggle_tf_method_visibility)
        self.main.radioButton_tfidf.toggled.connect(self.toggle_tf_method_visibility)
        self.main.radioButton_tfidfxnorm.toggled.connect(self.toggle_tf_method_visibility)
        self.main.radioButton_single.toggled.connect(self.toggle_input_mode_buttons)
        self.main.radioButton_batch.toggled.connect(self.toggle_input_mode_buttons)
        self.tf_method = "log"
        self.toggle_input_mode_buttons()
        self.toggle_tf_method_visibility()
        
        self.main.pushButton_process.setDisabled(True)
        self.main.pushButton_process.clicked.connect(self.handle_input_process)
        self.inverted_window = None
        self.main.pushButton_2.setDisabled(True)
        self.main.pushButton_2.clicked.connect(self.show_inverted_file_window)
        self.main.pushButton_1.clicked.connect(self.show_expanded_window)
        self.main.combobox.currentIndexChanged.connect(self.handle_tf_method)
        self.main.pushButton_relevant.clicked.connect(self.handle_relevant_input)
        
        self.main.radioButton_fullbert.clicked.connect(self.handle_bert_mode)
        self.main.radioButton_bertexp.clicked.connect(self.handle_bert_mode)

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
                lambda: self.handle_batch_input()
            )
            self.main.pushButton_folder.clicked.connect(
                lambda: self.handle_source_document_select()
            )
        else:
            self.main.pushButton_file.clicked.connect(
                lambda: self.handle_batch_input()
            )
            self.main.pushButton_folder.clicked.connect(
                lambda: self.handle_source_document_select()
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
        self.main.pushButton_file.setDisabled(is_single or not self.ready_to_process)
        self.main.pushButton_process.setDisabled(not is_single or not self.ready_to_process)
        
    def show_inverted_file_window(self):
        self.inverted_window = InvertedFileWindow()
        self.inverted_window.show()
    
    def show_expanded_window(self):
        self.expanded_window = ExpandedWindow()
        self.expanded_window.show()
        
    def handle_tf_method(self):
        if self.main.combobox.currentText() == "Logarithmic":
            self.tf_method = "log"
        elif self.main.combobox.currentText() == "Binary": 
            self.tf_method = "binary"
        elif self.main.combobox.currentText() == "Augmented":
            self.tf_method = "augmented"
        elif self.main.combobox.currentText() == "Raw":
            self.tf_method = "raw"
    
    def handle_bert_mode(self):
        global is_fullbert
        if self.main.radioButton_fullbert.isChecked():
            is_fullbert = True
        elif self.main.radioButton_bertexp.isChecked():
            is_fullbert = False
                    

    def handle_source_document_select(self):
        file_path, _ = QFileDialog.getOpenFileName(self.main, "Select Source Document", "", "All Files (*)")
        if file_path:
            tf = False
            idf = False
            normalize = False
            
            if self.main.radioButton_tf.isChecked():
                tf = True
            
            if self.main.radioButton_idf.isChecked():
                idf = True
            
            if self.main.radioButton_tfidf.isChecked():
                tf = True
                idf = True
            
            if self.main.radioButton_tfidfxnorm.isChecked():
                tf = True
                idf = True
                normalize = True
            
            original_process.process_source(
                file_path,
                self.main.checkBox_2.isChecked(),
                self.main.checkBox_1.isChecked(),
                tf = tf,
                idf = idf,
                normalize= normalize,
                scheme_tf = self.tf_method,
                scheme_idf="log"
            )
            
            expand_process.process_source(
                file_path,
                self.main.checkBox_2.isChecked(),
                self.main.checkBox_1.isChecked(),
                tf = tf,
                idf = idf,
                normalize= normalize,
                scheme_tf = self.tf_method,
                scheme_idf="log"
            )
            
            self.ready_to_process = True
            self.main.pushButton_2.setDisabled(False)
        else:
            self.ready_to_process = False
            self.main.pushButton_2.setDisabled(True)
    
    def handle_input_process(self):
        input_text = self.main.lineEdit.text().strip()
        if input_text:
            
            tf = False
            idf = False
            normalize = False
            
            if self.main.radioButton_tf.isChecked():
                tf = True
            
            if self.main.radioButton_idf.isChecked():
                idf = True
            
            if self.main.radioButton_tfidf.isChecked():
                tf = True
                idf = True
            
            if self.main.radioButton_tfidfxnorm.isChecked():
                tf = True
                idf = True
                normalize = True
            
            original_process.process_single_input(
                input_text=input_text,
                stop_word_elim=self.main.checkBox_2.isChecked(),
                stemming=self.main.checkBox_1.isChecked(),
                tf = tf,
                idf = idf,
                normalize= normalize,
                scheme_tf = self.tf_method,
                scheme_idf="log"
            )
            
            is_full_bert = self.main.radioButton_fullbert.isChecked()
            is_add_all = self.main.radioButton_1.isChecked()
            if is_add_all:
                num_of_added = -1
            else:
                num_of_added = self.main.spinBox_term_limit.value()
                            
            if is_full_bert:
                expand_process.bert_instant_single(
                    input_text=input_text,
                    stop_word_elim=self.main.checkBox_2.isChecked(),
                    stemming=self.main.checkBox_1.isChecked(),
                    tf = tf,
                    idf = idf,
                    normalize= normalize,
                    scheme_tf = self.tf_method,
                    scheme_idf="log",
                    num_of_added=num_of_added
                )
            else:
                expand_process.bert_expand_single(
                    input_text=input_text,
                    stop_word_elim=self.main.checkBox_2.isChecked(),
                    stemming=self.main.checkBox_1.isChecked(),
                    tf = tf,
                    idf = idf,
                    normalize= normalize,
                    scheme_tf = self.tf_method,
                    scheme_idf="log",
                    num_of_added=num_of_added
                )
            
    def handle_batch_input(self):
        file_path, _ = QFileDialog.getOpenFileName(self.main, "Select Input File", "", "All Files (*)")
        if file_path:
            tf = False
            idf = False
            normalize = False
            
            if self.main.radioButton_tf.isChecked():
                tf = True
            
            if self.main.radioButton_idf.isChecked():
                idf = True
            
            if self.main.radioButton_tfidf.isChecked():
                tf = True
                idf = True
            
            if self.main.radioButton_tfidfxnorm.isChecked():
                tf = True
                idf = True
                normalize = True
            
            original_process.process_batch_input(
                path_to_file=file_path,
                stop_word_elim=self.main.checkBox_2.isChecked(),
                stemming=self.main.checkBox_1.isChecked(),
                tf = tf,
                idf = idf,
                normalize= normalize,
                scheme_tf = self.tf_method,
                scheme_idf="log"
            )
            
            is_full_bert = self.main.radioButton_fullbert.isChecked()
            is_add_all = self.main.radioButton_1.isChecked()
            if is_add_all:
                num_of_added = -1
            else:
                num_of_added = self.main.spinBox_term_limit.value()
                            
            if is_full_bert:
                expand_process.bert_instant_batch(
                    path_to_file=file_path,
                    stop_word_elim=self.main.checkBox_2.isChecked(),
                    stemming=self.main.checkBox_1.isChecked(),
                    tf = tf,
                    idf = idf,
                    normalize= normalize,
                    scheme_tf = self.tf_method,
                    scheme_idf="log",
                    num_of_added=num_of_added
                )
            else:
                expand_process.bert_expand_batch(
                    path_to_file=file_path,
                    stop_word_elim=self.main.checkBox_2.isChecked(),
                    stemming=self.main.checkBox_1.isChecked(),
                    tf = tf,
                    idf = idf,
                    normalize= normalize,
                    scheme_tf = self.tf_method,
                    scheme_idf="log",
                    num_of_added=num_of_added
                )
            
    def handle_relevant_input(self):
        file_path, _ = QFileDialog.getOpenFileName(self.main, "Select Input File", "", "All Files (*)")
        if file_path:
            original_process.set_relevant(filepath=file_path)
            expand_process.set_relevant(filepath=file_path)
            
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
            self.ui = loader.load(ui_file, None)
            if self.ui is None:
                raise RuntimeError(f"Failed to load UI file: {ui_file}")
            self.setCentralWidget(self.ui)

        elif '--pyqt5' in sys.argv or '--pyqt6' in sys.argv:
            self.ui = uic.loadUi('inverted_window.ui')
            self.setCentralWidget(self.ui)

        else:
            raise RuntimeError("Unsupported Qt version.")
        self.ui.tableWidget_inverted.setRowCount(len(original_process.vocab))
        self.ui.spinBox_inverted.setMaximum(max(original_process.source_indices))
        self.ui.spinBox_inverted.setMinimum(min(original_process.source_indices))
        
        for row in range(len(original_process.vocab)):
            term = original_process.vocab[row]
            self.ui.tableWidget_inverted.setItem(row, 0, QTableWidgetItem(term))
            
        
        self.ui.pushButton_process_inverted.clicked.connect(self.show_inverted_file)
        
        
    def show_inverted_file(self):
        try:
            index = self.ui.spinBox_inverted.value()
            inverted = original_process.get_inverted(index)
            for row in range(len(original_process.vocab)):
                self.ui.tableWidget_inverted.setItem(row, 1, QTableWidgetItem(str(inverted[0][row])))
                self.ui.tableWidget_inverted.setItem(row, 2, QTableWidgetItem(str(inverted[1][row])))
                self.ui.tableWidget_inverted.setItem(row, 3, QTableWidgetItem(str(inverted[2][row])))
        except ValueError:
            pass
        
        

class ResultWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Retrieval Results")
        self.setAttribute(Qt.WA_DeleteOnClose, True)
        
        logo = QIcon("img/logo.png")
        self.setWindowIcon(logo)

        if '--pyside6' in sys.argv:
            loader = QUiLoader()
            ui_file = os.path.abspath("result_window.ui")
            self.ui = loader.load(ui_file, None)
            if self.ui is None:
                raise RuntimeError(f"Failed to load UI file: {ui_file}")
            self.setCentralWidget(self.ui)

        elif '--pyqt5' in sys.argv or '--pyqt6' in sys.argv:
            self.ui = uic.loadUi('result_window.ui')
            self.setCentralWidget(self.ui)

        else:
            raise RuntimeError("Unsupported Qt version.")
        
        self.ui.tableWidget_exp.setRowCount(len(original_process.source_indices))
        self.ui.tableWidget_exp_2.setRowCount(len(original_process.source_indices))
        self.ui.spinBox_retrieved.setMaximum(max(original_process.input_indices))
        self.ui.spinBox_retrieved.setMinimum(min(original_process.input_indices))
        self.ui.pushButton_process_retrieved.clicked.connect(self.show_result)
        self.ui.ori_value.setText(str(original_process.get_MAP()))
        self.ui.exp_value.setText(str(expand_process.get_MAP()))
        self.ui.export_original.clicked.connect(self.export_original_ranking)
        self.ui.export_expanded.clicked.connect(self.export_expanded_ranking)
    
    def show_result(self):
        index = self.ui.spinBox_retrieved.value()
        ranking = original_process.get_ranking(index)
        self.ui.ori_ap_value.setText(str(original_process.get_ap(index)))
        for row in range(len(original_process.source_indices)):
            self.ui.tableWidget_exp.setItem(row, 0, QTableWidgetItem(str(ranking[row][0])))
        
        
        self.ui.exp_ap_value.setText(str(expand_process.get_ap(index)))
        exp_ranking = []
        if is_fullbert:
            exp_ranking = expand_process.get_ranking_bert(index)
        else:
            exp_ranking = expand_process.get_ranking(index)
        for row in range(len(expand_process.source_indices)):
                self.ui.tableWidget_exp_2.setItem(row, 0, QTableWidgetItem(str(exp_ranking[row][0])))    
            
    def export_original_ranking(self):
        file_path, _ = QFileDialog.getSaveFileName(self, "Save Ranking", "", "Text Files (*.text)")
        if file_path:
            with open(file_path, 'w') as f:
                f.write(f"MAP.\n{original_process.get_MAP()}\n")
                for index in original_process.input_indices:
                    ranking = original_process.get_ranking(index)
                    f.write(".I\n")
                    f.write(f"{index}\n")
                    f.write(f"AP.\n{original_process.get_ap(index)}\n")
                    f.write(".X\n")
                    for doc_index, score in ranking:
                        f.write(f"{doc_index} {score}\n")
                    f.write("\n")
                    
    def export_expanded_ranking(self):
        file_path, _ = QFileDialog.getSaveFileName(self, "Save Expanded Ranking", "", "Text Files (*.text)")
        if file_path:
            with open(file_path, 'w') as f:
                f.write(f"MAP.\n{expand_process.get_MAP()}\n")
                for index in expand_process.input_indices:
                    ranking = []
                    if is_fullbert:
                        ranking = expand_process.get_ranking_bert(index)
                    else:
                        ranking = expand_process.get_ranking(index)
                    f.write(".I\n")
                    f.write(f"{index}\n")
                    f.write(f"AP.\n{expand_process.get_ap(index)}\n")
                    f.write(".X\n")
                    for doc_index, score in ranking:
                        f.write(f"{doc_index} {score}\n")
                    f.write("\n")
        
        
class ExpandedWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Expanded Query Results")
        self.setAttribute(Qt.WA_DeleteOnClose, True)
        
        logo = QIcon("img/logo.png")
        self.setWindowIcon(logo)

        if '--pyside6' in sys.argv:
            loader = QUiLoader()
            ui_file = os.path.abspath("queryexp_window.ui")
            self.ui = loader.load(ui_file, None)
            if self.ui is None:
                raise RuntimeError(f"Failed to load UI file: {ui_file}")
            self.setCentralWidget(self.ui)

        elif '--pyqt5' in sys.argv or '--pyqt6' in sys.argv:
            self.ui = uic.loadUi('queryexp_window.ui')
            self.setCentralWidget(self.ui)

        else:
            raise RuntimeError("Unsupported Qt version.")
        
        self.ui.pushButton.clicked.connect(self.show_result_window)
        
    def show_result_window(self):
        self.result_window = ResultWindow()
        self.result_window.show()
        self.close()
        
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
