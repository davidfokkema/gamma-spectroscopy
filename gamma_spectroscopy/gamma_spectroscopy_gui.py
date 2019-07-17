import csv
import ctypes
from math import floor
import sys
import time

import numpy as np
from pkg_resources import resource_filename

from PyQt5 import uic, QtWidgets, QtCore
import pyqtgraph as pg

from gamma_spectroscopy.picoscope_5000a import PicoScope5000A, INPUT_RANGES
from gamma_spectroscopy.fake_picoscope import FakePicoScope


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

    num_events = 0

    _is_running = False
    _trigger_channel = 'A'
    _is_trigger_enabled = False
    _is_upper_threshold_enabled = False
    _pulse_polarity = 'Positive'
    _polarity_sign = 1
    _is_baseline_correction_enabled = True

    _range = 0
    _offset_level = 0.
    _offset = 0.
    _threshold = 0.
    _upper_threshold = 1.
    _timebase = 0
    _pre_trigger_window = 0.
    _post_trigger_window = 0.
    _pre_samples = 0
    _post_samples = 0
    _num_samples = 0

    _t_start_run = 0

    def __init__(self):
        super().__init__()

        self._pulseheights = {'A': [], 'B': []}

        # self.scope = PicoScope5000A()
        self.scope = FakePicoScope()

        self.init_ui()

    def closeEvent(self, event):
        self._is_running = False
        self.scope.stop()

    def init_ui(self):
        pg.setConfigOption('background', 'w')
        pg.setConfigOption('foreground', 'k')

        ui_path = resource_filename('gamma_spectroscopy', 'gamma_spectroscopy_gui.ui')
        layout = uic.loadUi(ui_path, self)

        # Menubar
        menubar = QtWidgets.QMenuBar()

        export_spectrum_action = QtWidgets.QAction('&Export spectrum', self)
        export_spectrum_action.setShortcut('Ctrl+S')
        export_spectrum_action.triggered.connect(self.export_spectrum_dialog)

        file_menu = menubar.addMenu('&File')
        file_menu.addAction(export_spectrum_action)

        layout.setMenuBar(menubar)


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
        self.upper_threshold_box.valueChanged.connect(self.set_upper_threshold)
        self.trigger_box.stateChanged.connect(self.set_trigger_state)
        self.upper_trigger_box.stateChanged.connect(self.set_upper_trigger_state)
        self.trigger_channel_box.currentTextChanged.connect(self.set_trigger)
        self.timebase_box.valueChanged.connect(self.set_timebase)
        self.pre_trigger_box.valueChanged.connect(self.set_pre_trigger_window)
        self.post_trigger_box.valueChanged.connect(self.set_post_trigger_window)
        self.baseline_correction_box.stateChanged.connect(self.set_baseline_correction_state)

        self.lld_box.valueChanged.connect(self.update_spectrum_plot)
        self.uld_box.valueChanged.connect(self.update_spectrum_plot)
        self.num_bins_box.valueChanged.connect(self.update_spectrum_plot)

        self.clear_spectrum_button.clicked.connect(self.clear_spectrum)
        self.single_button.clicked.connect(self.start_scope_run)
        self.run_stop_button.clicked.connect(self.toggle_run_stop)

        self.run_timer.timeout.connect(self._update_run_time_label)

        self.init_event_plot()
        self.init_spectrum_plot()

        self._emit_value_changed_signal(self.offset_box)
        self._emit_value_changed_signal(self.threshold_box)
        self._emit_value_changed_signal(self.timebase_box)
        self._emit_value_changed_signal(self.pre_trigger_box)
        self._emit_value_changed_signal(self.post_trigger_box)

        self.show()

    def _emit_value_changed_signal(self, widget):
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
            self.num_events = 0
            self._update_run_time_label()
            self.run_timer.start()
            self.start_run_signal.emit()
            self.run_stop_button.setText("Stop")
            self.single_button.setDisabled(True)

    def stop_run(self):
            self._is_running = False
            self._update_run_time_label()
            self.scope.stop()
            self.run_timer.stop()
            self.run_stop_button.setText("Run")
            self.single_button.setDisabled(False)

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
        self.set_trigger()

    @QtCore.pyqtSlot(float)
    def set_offset(self, offset_level):
        self._offset_level = offset_level
        self._set_channel()
        self.set_trigger()

    @QtCore.pyqtSlot(float)
    def set_threshold(self, threshold):
        self._threshold = threshold
        self.set_trigger()

    @QtCore.pyqtSlot(float)
    def set_upper_threshold(self, threshold):
        self._upper_threshold = threshold

    @QtCore.pyqtSlot(int)
    def set_trigger_state(self, state):
        self._is_trigger_enabled = state
        self.set_trigger()

    @QtCore.pyqtSlot(int)
    def set_upper_trigger_state(self, state):
        self._is_upper_threshold_enabled = state

    @QtCore.pyqtSlot(int)
    def set_polarity(self, idx):
        self._pulse_polarity = self.POLARITY[idx]
        self._polarity_sign = self.POLARITY_SIGN[idx]
        self.set_trigger()

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

    def set_trigger(self):
        edge = 'RISING' if self._pulse_polarity == 'Positive' else 'FALLING'
        self.scope.stop()
        # get last letter of trigger channel box ('Channel A' -> 'A')
        channel = self.trigger_channel_box.currentText()[-1]
        self._trigger_channel = channel
        self.scope.set_trigger(channel, self._polarity_sign * self._threshold,
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
        self.num_events_label.setText(f"({self.num_events} events)")

    @QtCore.pyqtSlot()
    def fetch_data(self):
        t, [A, B] = self.scope.get_data()
        if A is not None:
            self.num_events += len(A)
            self.plot_data_signal.emit({'x': t, 'A': A, 'B': B})
        if self._is_running:
            self.start_run_signal.emit()
        self.check_run_time()

    def check_run_time(self):
        run_time = time.time() - self._t_start_run
        if run_time >= self.run_duration_box.value():
            self.stop_run()

    @QtCore.pyqtSlot()
    def clear_spectrum(self):
        self._t_start_run = time.time()
        self.num_events = 0
        self._pulseheights = {'A': [], 'B': []}
        self._update_run_time_label()
        self.init_spectrum_plot()

    @QtCore.pyqtSlot(dict)
    def plot_data(self, data):
        x, A, B = data['x'], data['A'], data['B']

        pulseheights = []
        for data in A, B:
            data *= self._polarity_sign
            if self._is_baseline_correction_enabled:
                num_samples = int(self._pre_samples * .8)
                correction = data[:, :num_samples].mean(axis=1)
            else:
                correction = 0
            ph = (data.max(axis=1) - correction) * 1e3
            pulseheights.append(ph)

        pulseheights = np.array(pulseheights)
        if self._is_upper_threshold_enabled:
            channel_idx = ['A', 'B'].index(self._trigger_channel)
            pulseheights = pulseheights.compress(
                pulseheights[channel_idx, :] <= self._upper_threshold * 1e3,
                axis=1)

        for channel, values in zip(['A', 'B'], pulseheights):
            self._pulseheights[channel].extend(values)

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
        if self.ch_A_enabled_box.isChecked():
            self.event_plot.plot(x * 1e6, A[-1], pen={'color': 'k', 'width': 4.})
        if self.ch_B_enabled_box.isChecked():
            self.event_plot.plot(x * 1e6, B[-1], pen={'color': 'b', 'width': 4.})

    def init_spectrum_plot(self):
        self.spectrum_plot.clear()
        self.spectrum_plot.setLabels(title='Spectrum',
                                     bottom='Pulseheight [mV]', left='Counts')

    def update_spectrum_plot(self):
        self.spectrum_plot.clear()
        x, bins, channel_counts = self.make_spectrum()
        for counts, color in zip(channel_counts, ['k', 'b']):
            if counts is not None:
                self.spectrum_plot.plot(x, counts, pen={'color': color,
                                                        'width': 4.})
        self.spectrum_plot.setXRange(0, 2 * self._range * 1e3)

    def make_spectrum(self):
        xrange = 2 * self._range * 1e3
        xmin = .01 * self.lld_box.value() * xrange
        xmax = .01 * self.uld_box.value() * xrange
        bins = np.linspace(xmin, xmax, self.num_bins_box.value())
        x = (bins[:-1] + bins[1:]) / 2
        channel_counts = []

        for channel in 'A', 'B':
            box = getattr(self, f'ch_{channel}_enabled_box')
            if box.isChecked():
                n, _ = np.histogram(self._pulseheights[channel], bins=bins)
                channel_counts.append(n)
            else:
                channel_counts.append(None)

        return x, bins, channel_counts

    def export_spectrum_dialog(self):
        """Dialog for exporting a data file."""

        file_path, _ = QtWidgets.QFileDialog.getSaveFileName(
            self, caption="Save spectrum", directory="spectrum.csv")

        x, _, channel_counts = self.make_spectrum()
        channel_counts = [u if u is not None else [0] * len(x) for
                          u in channel_counts]

        with open(file_path, 'w') as f:
            writer = csv.writer(f)
            writer.writerow(('pulseheight', 'counts_ch_A', 'counts_ch_B'))
            for row in zip(x, *channel_counts):
                writer.writerow(row)


def main():
    qtapp = QtWidgets.QApplication(sys.argv)
    ui = UserInterface()
    sys.exit(qtapp.exec_())


if __name__ == '__main__':
    main()
