import ctypes
import os.path
import sys

from PyQt5 import uic, QtWidgets, QtCore
import pyqtgraph as pg

from picoscope_5000a import PicoScope5000A


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

        pg.setConfigOption('background', 'w')
        pg.setConfigOption('foreground', 'k')

        ui_path = 'gamma_spectroscopy_gui.ui'
        uic.loadUi(ui_path, self)

        self.timer = QtCore.QTimer()
        self.timer.timeout.connect(self.start_run)

        self.new_data_signal.connect(self.fetch_data)
        self.callback = create_callback(self.new_data_signal)

        self.plot_data_signal.connect(self.plot_data)

        self.scope = PicoScope5000A()
        self.scope.set_channel('A', 'DC', 0.5)

        self.timer.start(100)

        self.show()

    @QtCore.pyqtSlot()
    def start_run(self):
        if self._is_running:
            return
        else:
            self._is_running = True
            self.scope.set_up_buffers(200)
            self.scope.start_run(100, 100, 20, callback=self.callback)

    @QtCore.pyqtSlot()
    def fetch_data(self):
        t, data = self.scope.get_data()
        self._is_running = False
        self.plot_data_signal.emit({'x': t, 'y': data[0]})

    @QtCore.pyqtSlot(dict)
    def plot_data(self, data):
        self.plot.clear()
        self.plot.plot(data['x'] * 1e6, data['y'] * 1e3, pen='k')
        self.plot.setLabels(title='Scintillator event', bottom='Time [us]',
                            left='Signal [mV]')
        # self.plot.setYRange(0, -500)


if __name__ == '__main__':
    qtapp = QtWidgets.QApplication(sys.argv)
    ui = UserInterface()
    sys.exit(qtapp.exec_())
