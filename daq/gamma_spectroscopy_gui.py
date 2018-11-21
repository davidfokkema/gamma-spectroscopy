import ctypes
import os.path
import sys

from PyQt5 import uic, QtWidgets, QtCore
import pyqtgraph

from picoscope_5000a import PicoScope5000A


def create_callback(signal):
    @ctypes.CFUNCTYPE(None, ctypes.c_int16, ctypes.c_int, ctypes.c_void_p)
    def my_callback(handle, status, parameters):
        print("callback")
        signal.emit()
    return my_callback


class UserInterface(QtWidgets.QMainWindow):

    new_data_signal = QtCore.pyqtSignal()

    def __init__(self):
        super().__init__()

        ui_path = 'gamma_spectroscopy_gui.ui'
        uic.loadUi(ui_path, self)

        self.timer = QtCore.QTimer()
        self.timer.timeout.connect(self.start_run)

        self.new_data_signal.connect(self.fetch_data)
        self.callback = create_callback(self.new_data_signal)

        self.scope = PicoScope5000A()
        self.scope.set_channel('A', 'DC', 50e-3)

        self.timer.start(1000)

        self.show()

    @QtCore.pyqtSlot()
    def start_run(self):
        print("Starting run...")
        self.scope._ps_run_block(100, 100, 4, self.callback)

    @QtCore.pyqtSlot()
    def fetch_data(self):
        print("Fetching data...")


if __name__ == '__main__':
    qtapp = QtWidgets.QApplication(sys.argv)
    ui = UserInterface()
    sys.exit(qtapp.exec_())
