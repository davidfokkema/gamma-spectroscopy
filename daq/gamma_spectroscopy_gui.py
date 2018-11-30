import ctypes
import os.path
import sys
import time

from PyQt5 import uic, QtWidgets, QtCore
import pyqtgraph as pg

from picoscope_5000a import PicoScope5000A, INPUT_RANGES


def create_callback(signal):
    @ctypes.CFUNCTYPE(None, ctypes.c_int16, ctypes.c_int, ctypes.c_void_p)
    def my_callback(handle, status, parameters):
        signal.emit()
    return my_callback


class UserInterface(QtWidgets.QMainWindow):

    start_run_signal = QtCore.pyqtSignal()
    new_data_signal = QtCore.pyqtSignal()
    plot_data_signal = QtCore.pyqtSignal(dict)

    _is_running = False
    _is_trigger_enabled = False

    _range = 0
    _offset = 0.
    _threshold = 0.


    def __init__(self):
        super().__init__()

        self.scope = PicoScope5000A()

        self.init_ui()

    def init_ui(self):
        pg.setConfigOption('background', 'w')
        pg.setConfigOption('foreground', 'k')

        ui_path = 'gamma_spectroscopy_gui.ui'
        uic.loadUi(ui_path, self)

        self.start_run_signal.connect(self.start_run)

        self.new_data_signal.connect(self.fetch_data)
        self.callback = create_callback(self.new_data_signal)

        self.plot_data_signal.connect(self.plot_data)

        self.range_box.addItems(INPUT_RANGES.values())
        self.range_box.currentIndexChanged.connect(self.set_range)
        self.range_box.setCurrentIndex(6)
        self.offset_box.valueChanged.connect(self.set_offset)
        self.threshold_box.valueChanged.connect(self.set_threshold)
        self.trigger_box.stateChanged.connect(self.set_trigger_state)

        self.run_stop_button.clicked.connect(self.toggle_run_stop)

        self.plot_data_signal.emit({})

        self.show()

    @QtCore.pyqtSlot()
    def toggle_run_stop(self):
        if not self._is_running:
            self._is_running = True
            self.start_run_signal.emit()
            self.run_stop_button.setText("Stop")
        else:
            self._is_running = False
            self.scope.stop()
            self.run_stop_button.setText("Run")

    @QtCore.pyqtSlot()
    def start_run(self):
        self.scope.set_up_buffers(2000)
        self.scope.start_run(1000, 1000, 2000, callback=self.callback)

    @QtCore.pyqtSlot(int)
    def set_range(self, range_idx):
        self.scope.stop()
        ranges = list(INPUT_RANGES.keys())
        range = ranges[range_idx]
        self.scope.set_channel('A', 'DC', range)
        self.scope.set_trigger('A', self._threshold, 'FALLING', is_enabled=self._is_trigger_enabled)
        self._range = range

    @QtCore.pyqtSlot(float)
    def set_offset(self, offset):
        self.scope.stop()
        self.scope.set_channel('A', 'DC', self._range, offset)
        self.scope.set_trigger('A', self._threshold, 'FALLING', is_enabled=self._is_trigger_enabled)
        self._offset = offset

    @QtCore.pyqtSlot(float)
    def set_threshold(self, threshold):
        self.scope.stop()
        self.scope.set_trigger('A', threshold, 'FALLING', is_enabled=self._is_trigger_enabled)
        self._threshold = threshold

    @QtCore.pyqtSlot(int)
    def set_trigger_state(self, state):
        self.scope.stop()
        self.scope.set_trigger('A', self._threshold, 'FALLING', is_enabled=state)
        self._is_trigger_enabled = state

    @QtCore.pyqtSlot()
    def fetch_data(self):
        t, data = self.scope.get_data()
        if data is not None:
            self.plot_data_signal.emit({'x': t, 'y': data[0]})
        if self._is_running:
            self.start_run_signal.emit()

    @QtCore.pyqtSlot(dict)
    def plot_data(self, data):
        self.plot.clear()
        if data:
            self.plot.plot(data['x'] * 1e6, data['y'], pen='k')
        self.plot.setLabels(title='Scintillator event', bottom='Time [us]',
                            left='Signal [mV]')
        self.plot.setYRange(-self._range - self._offset, self._range - self._offset)


if __name__ == '__main__':
    qtapp = QtWidgets.QApplication(sys.argv)
    ui = UserInterface()
    sys.exit(qtapp.exec_())
