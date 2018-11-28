import ctypes
import os.path
import sys

from PyQt5 import uic, QtWidgets, QtCore
import pyqtgraph as pg

from picoscope_5000a import PicoScope5000A, INPUT_RANGES


def create_callback(signal):
    @ctypes.CFUNCTYPE(None, ctypes.c_int16, ctypes.c_int, ctypes.c_void_p)
    def my_callback(handle, status, parameters):
        signal.emit()
    return my_callback


class UserInterface(QtWidgets.QMainWindow):

    new_data_signal = QtCore.pyqtSignal()
    plot_data_signal = QtCore.pyqtSignal(dict)

    _is_running = False

    def __init__(self):
        super().__init__()

        self.scope = PicoScope5000A()
        self.scope.set_channel('A', 'DC', .5)

        self.init_ui()

        self.timer = QtCore.QTimer()
        self.timer.timeout.connect(self.start_run)
        self.timer.start(100)

    def init_ui(self):
        pg.setConfigOption('background', 'w')
        pg.setConfigOption('foreground', 'k')

        ui_path = 'gamma_spectroscopy_gui.ui'
        uic.loadUi(ui_path, self)

        self.new_data_signal.connect(self.fetch_data)
        self.callback = create_callback(self.new_data_signal)

        self.plot_data_signal.connect(self.plot_data)

        self.range_box.addItems(INPUT_RANGES.values())
        self.range_box.currentIndexChanged.connect(self.set_range)
        self.range_box.setCurrentIndex(6)

        self.show()

    @QtCore.pyqtSlot()
    def start_run(self):
        if self._is_running:
            return
        else:
            self._is_running = True
            self.scope.set_up_buffers(20000)
            self.scope.start_run(10000, 10000, 20, callback=self.callback)

    @QtCore.pyqtSlot(int)
    def set_range(self, range_idx):
        print(f"RANGE CHANGED to {range_idx}")
        ranges = list(INPUT_RANGES.keys())
        range = ranges[range_idx]
        self.scope.set_channel('A', 'DC', range)
        self._range = range

    @QtCore.pyqtSlot()
    def fetch_data(self):
        t, data = self.scope.get_data()
        self._is_running = False
        self.plot_data_signal.emit({'x': t, 'y': data[0]})

    @QtCore.pyqtSlot(dict)
    def plot_data(self, data):
        self.plot.clear()
        self.plot.plot(data['x'] * 1e6, data['y'], pen='k')
        self.plot.setLabels(title='Scintillator event', bottom='Time [us]',
                            left='Signal [mV]')
        self.plot.setYRange(-self._range, self._range)


if __name__ == '__main__':
    qtapp = QtWidgets.QApplication(sys.argv)
    ui = UserInterface()
    sys.exit(qtapp.exec_())
