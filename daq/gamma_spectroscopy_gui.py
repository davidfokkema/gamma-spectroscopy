import ctypes
from math import floor
import sys
import time

import numpy as np

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
    _timebase = 0
    _pre_trigger_window = 0.
    _post_trigger_window = 0.
    _pre_samples = 0
    _post_samples = 0
    _num_samples = 0

    _t_last_plot_update = 0


    def __init__(self):
        super().__init__()

        self._pulseheights = []

        self.scope = PicoScope5000A()

        self.init_ui()

    def closeEvent(self, event):
        self._is_running = False
        self.scope.stop()

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
        self.timebase_box.valueChanged.connect(self.set_timebase)
        self.pre_trigger_box.valueChanged.connect(self.set_pre_trigger_window)
        self.post_trigger_box.valueChanged.connect(self.set_post_trigger_window)

        self.clear_spectrum_button.clicked.connect(self.clear_spectrum)
        self.run_stop_button.clicked.connect(self.toggle_run_stop)

        self.plot_data_signal.emit({})

        self._trigger_value_changed_signal(self.offset_box)
        self._trigger_value_changed_signal(self.threshold_box)
        self._trigger_value_changed_signal(self.timebase_box)
        self._trigger_value_changed_signal(self.pre_trigger_box)
        self._trigger_value_changed_signal(self.post_trigger_box)

        self.show()

    def _trigger_value_changed_signal(self, widget):
        widget.valueChanged.emit(widget.value())

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
        self.scope.set_up_buffers(self._num_samples)
        self.scope.start_run(self._pre_samples, self._post_samples,
                             self._timebase, callback=self.callback)

    @QtCore.pyqtSlot(int)
    def set_range(self, range_idx):
        ranges = list(INPUT_RANGES.keys())
        self._range = ranges[range_idx]
        self._set_channel()
        self._set_trigger()

    @QtCore.pyqtSlot(float)
    def set_offset(self, offset):
        self._offset = offset
        self._set_channel()
        self._set_trigger()

    @QtCore.pyqtSlot(float)
    def set_threshold(self, threshold):
        self._threshold = threshold
        self._set_trigger()

    @QtCore.pyqtSlot(int)
    def set_trigger_state(self, state):
        self.scope.stop()
        self.scope.set_trigger('A', self._threshold, 'FALLING', is_enabled=state)
        self._is_trigger_enabled = state

    def _set_channel(self):
        self.scope.stop()
        self.scope.set_channel('A', 'DC', self._range, self._offset)

    def _set_trigger(self):
        self.scope.stop()
        self.scope.set_trigger('A', self._threshold, 'FALLING', is_enabled=self._is_trigger_enabled)

    @QtCore.pyqtSlot(int)
    def set_timebase(self, timebase):
        self._timebase = timebase
        dt = self.scope.get_interval_from_timebase(timebase)
        self.sampling_time_label.setText(f"{dt / 1e3:.3f} μs")
        self._update_num_samples()

    @QtCore.pyqtSlot(float)
    def set_pre_trigger_window(self, pre_trigger_window):
        self._pre_trigger_window = pre_trigger_window * 1e3
        self._update_num_samples()

    @QtCore.pyqtSlot(float)
    def set_post_trigger_window(self, post_trigger_window):
        self._post_trigger_window = post_trigger_window * 1e3
        self._update_num_samples()

    def _update_num_samples(self):
        pre_samples, post_samples = self._calculate_num_samples()
        num_samples = pre_samples + post_samples
        self.num_samples_label.setText(str(num_samples))

        self._pre_samples = pre_samples
        self._post_samples = post_samples
        self._num_samples = num_samples

    def _calculate_num_samples(self):
        dt = self.scope.get_interval_from_timebase(self._timebase)
        pre_samples = floor(self._pre_trigger_window / dt)
        post_samples = floor(self._post_trigger_window / dt) + 1
        return pre_samples, post_samples

    @QtCore.pyqtSlot()
    def fetch_data(self):
        t, data = self.scope.get_data()
        if data is not None:
            self.plot_data_signal.emit({'x': t, 'y': data[0]})
        if self._is_running:
            self.start_run_signal.emit()

    @QtCore.pyqtSlot()
    def clear_spectrum(self):
        self._pulseheights = []

    @QtCore.pyqtSlot(dict)
    def plot_data(self, data):
        t = time.time()
        interval = 1 / self.plot_limit_box.value()
        if t - self._t_last_plot_update > interval:
            self._t_last_plot_update = t
            self.update_event_plot(data)
            self.update_spectrum_plot(data)

    def update_event_plot(self, data):
        self.event_plot.clear()
        if data:
            self.event_plot.plot(data['x'] * 1e6, data['y'], pen='k')
        self.event_plot.setLabels(title='Scintillator event', bottom='Time [us]',
                            left='Signal [mV]')
        self.event_plot.setYRange(-self._range - self._offset, self._range - self._offset)

    def update_spectrum_plot(self, data):
        self.spectrum_plot.setLabels(title='Spectrum',
                                     bottom='Pulseheight [mV]', left='Counts')
        self.spectrum_plot.setXRange(0, 2 * self._range * 1e3)

        if data:
            pulseheight = (-data['y']).max() * 1e3
            self._pulseheights.append(pulseheight)
            n, bins = np.histogram(self._pulseheights, bins=100)
            x = (bins[:-1] + bins[1:]) / 2
            self.spectrum_plot.clear()
            self.spectrum_plot.plot(x, n)


if __name__ == '__main__':
    qtapp = QtWidgets.QApplication(sys.argv)
    ui = UserInterface()
    sys.exit(qtapp.exec_())
