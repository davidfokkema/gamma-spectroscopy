import sys

from PyQt5 import QtWidgets, QtCore


class UserInterface(QtWidgets.QWidget):

    def __init__(self):
        super().__init__()

        self.timer = QtCore.QTimer()
        self.timer.timeout.connect(self.fetch_data)
        self.timer.start(1000)

    @QtCore.pyqtSlot()
    def fetch_data(self):
        print("Fetching data...")


if __name__ == '__main__':
    qtapp = QtWidgets.QApplication(sys.argv)
    ui = UserInterface()
    sys.exit(qtapp.exec_())
