from PyQt6 import QtWidgets, uic
from PyQt6.QtWidgets import QFileDialog
import sys
import os

class MyWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super(MyWindow, self).__init__()    
        uic.loadUi("windows/gui.ui", self)
        
        self.setFixedSize(700, 400)

        self.radioButton.toggled.connect(self.handle_radio_selection)
        self.radioButton_2.toggled.connect(self.handle_radio_selection)

        self.pushButton.clicked.connect(self.open_file_dialog)
        self.pushButton_2.clicked.connect(self.open_file_dialog2)

        self.pushButton.hide()
        self.label_3.hide()
        
        self.weightButton_1.toggled.connect(self.handle_weight_selection)
        self.weightButton_2.toggled.connect(self.handle_weight_selection)
        self.weightButton_3.toggled.connect(self.handle_weight_selection)
        self.weightButton_4.toggled.connect(self.handle_weight_selection)
        
        self.spinBox.setMinimum(0)
        self.spinBox.hide()
        self.buttonExp_1.toggled.connect(self.handle_expansion_selection)
        self.buttonExp_2.toggled.connect(self.handle_expansion_selection)
        
        self.buttonInverted.hide()
        

    def handle_radio_selection(self):
        if self.radioButton.isChecked():
            self.lineEdit.show()
            self.pushButton.hide()
            self.label_3.hide()
        elif self.radioButton_2.isChecked():
            self.lineEdit.hide()
            self.pushButton.show()
            self.label_3.show()

    def open_file_dialog(self):
        file_name, _ = QFileDialog.getOpenFileName(self, "Open File", "", "All Files (*)")
        if file_name:
            self.label_3.setText(f"Selected file: {os.path.basename(file_name)}")
            
    def open_file_dialog2(self):
        file_name, _ = QFileDialog.getOpenFileName(self, "Open File", "", "All Files (*)")
        if file_name:
            self.label_7.setText(f"Selected file: {os.path.basename(file_name)}")
            self.buttonInverted.show()
            
    def handle_weight_selection(self):
        if self.weightButton_2.isChecked():
            self.label_6.hide()
            self.comboBox.hide()
        else:
            self.label_6.show()
            self.comboBox.show()
            
    def handle_expansion_selection(self):
        if self.buttonExp_1.isChecked():
            self.spinBox.hide()
        else:
            self.spinBox.show()

app = QtWidgets.QApplication(sys.argv)
window = MyWindow()
window.show()
sys.exit(app.exec())