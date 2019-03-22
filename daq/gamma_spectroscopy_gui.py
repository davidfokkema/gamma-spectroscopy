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

    POLARITY = ['Positive', 'Negative']
    POLARITY_SIGN = [1, -1]

    start_run_signal = QtCore.pyqtSignal()
    new_data_signal = QtCore.pyqtSignal()
    plot_data_signal = QtCore.pyqtSignal(dict)

    run_timer = QtCore.QTimer(interval=1000)

    _is_running = False
    _is_trigger_enabled = False
    _pulse_polarity = 'Positive'
    _polarity_sign = 1
    _is_baseline_correction_enabled = True

    _range = 0
    _offset_level = 0.
    _offset = 0.
    _threshold = 0.
    _timebase = 0
    _pre_trigger_window = 0.
    _post_trigger_window = 0.
    _pre_samples = 0
    _post_samples = 0
    _num_samples = 0

    _t_last_plot_update = 0
    _t_start_run = 0


    def __init__(self):
        super().__init__()

        self._pulseheights = {'A': [], 'B': []}

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

        self.start_run_signal.connect(self.start_scope_run)

        self.new_data_signal.connect(self.fetch_data)
        self.callback = create_callback(self.new_data_signal)

        self.plot_data_signal.connect(self.plot_data)

        self.range_box.addItems(INPUT_RANGES.values())
        self.range_box.currentIndexChanged.connect(self.set_range)
        self.range_box.setCurrentIndex(6)
        self.polarity_box.addItems(self.POLARITY)
        self.polarity_box.currentIndexChanged.connect(self.set_polarity)
        self._pulse_polarity = self.POLARITY[0]
        self.offset_box.valueChanged.connect(self.set_offset)
        self.threshold_box.valueChanged.connect(self.set_threshold)
        self.trigger_box.stateChanged.connect(self.set_trigger_state)
        self.timebase_box.valueChanged.connect(self.set_timebase)
        self.pre_trigger_box.valueChanged.connect(self.set_pre_trigger_window)
        self.post_trigger_box.valueChanged.connect(self.set_post_trigger_window)
        self.baseline_correction_box.stateChanged.connect(self.set_baseline_correction_state)

        self.clear_spectrum_button.clicked.connect(self.clear_spectrum)
        self.run_stop_button.clicked.connect(self.toggle_run_stop)

        self.run_timer.timeout.connect(self._update_run_time_label)

        self.init_event_plot()
        self.init_spectrum_plot()

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
            self.start_run()
        else:
            self.stop_run()

    def start_run(self):
            self._is_running = True
            self.clear_spectrum()
            self._t_start_run = time.time()
            self._update_run_time_label()
            self.run_timer.start()
            self.start_run_signal.emit()
            self.run_stop_button.setText("Stop")

    def stop_run(self):
            self._is_running = False
            self._update_run_time_label()
            self.scope.stop()
            self.run_timer.stop()
            self.run_stop_button.setText("Run")

    @QtCore.pyqtSlot()
    def start_scope_run(self):
        num_captures = self.num_captures_box.value()
        self.scope.set_up_buffers(self._num_samples, num_captures)
        self.scope.start_run(self._pre_samples, self._post_samples,
                             self._timebase, num_captures,
                             callback=self.callback)

    @QtCore.pyqtSlot(int)
    def set_range(self, range_idx):
        ranges = list(INPUT_RANGES.keys())
        self._range = ranges[range_idx]
        self._set_channel()
        self._set_trigger()

    @QtCore.pyqtSlot(float)
    def set_offset(self, offset_level):
        self._offset_level = offset_level
        self._set_channel()
        self._set_trigger()

    @QtCore.pyqtSlot(float)
    def set_threshold(self, threshold):
        self._threshold = threshold
        self._set_trigger()

    @QtCore.pyqtSlot(int)
    def set_trigger_state(self, state):
        self._is_trigger_enabled = state
        self._set_trigger()

    @QtCore.pyqtSlot(int)
    def set_polarity(self, idx):
        self._pulse_polarity = self.POLARITY[idx]
        self._polarity_sign = self.POLARITY_SIGN[idx]
        self._set_trigger()

    @QtCore.pyqtSlot(int)
    def set_baseline_correction_state(self, state):
        self._is_baseline_correction_enabled = state

    def _set_channel(self):
        self.scope.stop()
        self._offset = np.interp(self._offset_level, [-100, 100],
                                 [-self._range, self._range])
        self.scope.set_channel('A', 'DC', self._range,
                               self._polarity_sign * self._offset)
        self.scope.set_channel('B', 'DC', self._range,
                               self._polarity_sign * self._offset)
        self.event_plot.setYRange(-self._range - self._offset,
                                  self._range - self._offset)

    def _set_trigger(self):
        edge = 'RISING' if self._pulse_polarity == 'Positive' else 'FALLING'
        self.scope.stop()
        self.scope.set_trigger('A', self._polarity_sign * self._threshold,
                               edge, is_enabled=self._is_trigger_enabled)

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

    def _update_run_time_label(self):
        run_time = round(time.time() - self._t_start_run)
        self.run_time_label.setText(f"{run_time} s")

    @QtCore.pyqtSlot()
    def fetch_data(self):
        self.check_run_time()
        t, [A, B] = self.scope.get_data()
        if A is not None:
            self.plot_data_signal.emit({'x': t, 'A': A, 'B': B})
        if self._is_running:
            self.start_run_signal.emit()

    def check_run_time(self):
        run_time = time.time() - self._t_start_run
        if run_time >= self.run_duration_box.value():
            self.stop_run()

    @QtCore.pyqtSlot()
    def clear_spectrum(self):
        self._t_start_run = time.time()
        self._update_run_time_label()
        self._pulseheights = {'A': [], 'B': []}
        self.init_spectrum_plot()

    @QtCore.pyqtSlot(dict)
    def plot_data(self, data):
        x, A, B = data['x'], data['A'], data['B']

        for data, pulseheights in [(A, self._pulseheights['A']),
                                   (B, self._pulseheights['B'])]:
            data *= self._polarity_sign
            if self._is_baseline_correction_enabled:
                num_samples = int(self._pre_samples * .8)
                correction = data[:, :num_samples].mean(axis=1)
            else:
                correction = 0
            ph = (data.max(axis=1) - correction) * 1e3
            pulseheights.extend(ph)

        t = time.time()
        interval = 1 / self.plot_limit_box.value()
        if t - self._t_last_plot_update > interval:
            self._t_last_plot_update = t
            self.update_event_plot(x, A, B)
            self.update_spectrum_plot()

    def init_event_plot(self):
        self.event_plot.clear()
        self.event_plot.setLabels(title='Scintillator event',
                                  bottom='Time [us]', left='Signal [V]')
        self.event_plot.setYRange(-self._range - self._offset,
                                  self._range - self._offset)

    def update_event_plot(self, x, A, B):
        self.event_plot.clear()
        self.event_plot.plot(x * 1e6, A[-1], pen={'color': 'k', 'width': 2.})
        self.event_plot.plot(x * 1e6, B[-1], pen={'color': 'b', 'width': 2.})

    def init_spectrum_plot(self):
        self.spectrum_plot.clear()
        self.spectrum_plot.setLabels(title='Spectrum',
                                     bottom='Pulseheight [mV]', left='Counts')

    def update_spectrum_plot(self):
        self.spectrum_plot.clear()
        for channel, color in [('A', 'k'), ('B', 'b')]:
            xmin, xmax = 0, 2 * self._range * 1e3
            bins = np.linspace(xmin, xmax, 100)
            n, bins = np.histogram(self._pulseheights[channel], bins=bins)
            x = (bins[:-1] + bins[1:]) / 2
            self.spectrum_plot.plot(x, n, pen={'color': color, 'width': 2.})
        self.spectrum_plot.setXRange(0, 2 * self._range * 1e3)


if __name__ == '__main__':
    qtapp = QtWidgets.QApplication(sys.argv)
    ui = UserInterface()
    sys.exit(qtapp.exec_())
