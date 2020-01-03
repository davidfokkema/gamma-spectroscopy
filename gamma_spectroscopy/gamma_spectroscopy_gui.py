import argparse
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


GUIDE_COLORS = {
    'red': (255, 0, 0, 63),
    'green': (0, 255, 0, 63),
    'blue': (0, 0, 255, 63),
    'purple': (255, 0, 255, 63),
}


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
    _t_prev_run_time = 0
    _run_min_baseline = [0, 0]
    _run_max_baseline = [0, 0]

    def __init__(self, use_fake=False):
        super().__init__()

        self._pulseheights = {'A': [], 'B': []}

        if use_fake:
            self.scope = FakePicoScope()
        else:
            self.scope = PicoScope5000A()

        self.init_ui()

    def closeEvent(self, event):
        self._is_running = False
        self.scope.stop()

    def init_ui(self):
        pg.setConfigOption('background', 'w')
        pg.setConfigOption('foreground', 'k')
        pg.setConfigOption('antialias', True)

        ui_path = resource_filename('gamma_spectroscopy',
                                    'gamma_spectroscopy_gui.ui')
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
        self.upper_trigger_box.stateChanged.connect(
            self.set_upper_trigger_state)
        self.trigger_channel_box.currentTextChanged.connect(self.set_trigger)
        self.timebase_box.valueChanged.connect(self.set_timebase)
        self.pre_trigger_box.valueChanged.connect(self.set_pre_trigger_window)
        self.post_trigger_box.valueChanged.connect(
            self.set_post_trigger_window)
        self.baseline_correction_box.stateChanged.connect(
            self.set_baseline_correction_state)

        self.lld_box.valueChanged.connect(self.update_spectrum_plot)
        self.uld_box.valueChanged.connect(self.update_spectrum_plot)
        self.num_bins_box.valueChanged.connect(self.update_spectrum_plot)

        self.clear_run_button.clicked.connect(self.clear_run)
        self.single_button.clicked.connect(self.start_scope_run)
        self.run_stop_button.clicked.connect(self.toggle_run_stop)

        self.reset_event_axes_button.clicked.connect(self.reset_event_axes)
        self.reset_spectrum_axes_button.clicked.connect(
            self.reset_spectrum_axes)

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
        self._t_start_run = time.time()
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
        run_time = time.time() - self._t_start_run
        self._t_prev_run_time += run_time
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
        self.set_channel()
        self.set_trigger()

    @QtCore.pyqtSlot(float)
    def set_offset(self, offset_level):
        self._offset_level = offset_level
        self.set_channel()
        self.set_trigger()

    @QtCore.pyqtSlot(float)
    def set_threshold(self, threshold):
        self._threshold = threshold
        self.set_trigger()

    @QtCore.pyqtSlot(float)
    def set_upper_threshold(self, threshold):
        self._upper_threshold = threshold
        self.scope.stop()

    @QtCore.pyqtSlot(int)
    def set_trigger_state(self, state):
        self._is_trigger_enabled = state
        self.set_trigger()

    @QtCore.pyqtSlot(int)
    def set_upper_trigger_state(self, state):
        self._is_upper_threshold_enabled = state
        self.scope.stop()

    @QtCore.pyqtSlot(int)
    def set_polarity(self, idx):
        self._pulse_polarity = self.POLARITY[idx]
        self._polarity_sign = self.POLARITY_SIGN[idx]
        self.set_channel()
        self.set_trigger()

    @QtCore.pyqtSlot(int)
    def set_baseline_correction_state(self, state):
        self._is_baseline_correction_enabled = state

    def set_channel(self):
        self._offset = np.interp(self._offset_level, [-100, 100],
                                 [-self._range, self._range])
        self.scope.set_channel('A', 'DC', self._range,
                               self._polarity_sign * self._offset)
        self.scope.set_channel('B', 'DC', self._range,
                               self._polarity_sign * self._offset)
        self.event_plot.setYRange(-self._range - self._offset,
                                  self._range - self._offset)
        self.scope.stop()

    def set_trigger(self):
        edge = 'RISING' if self._pulse_polarity == 'Positive' else 'FALLING'
        # get last letter of trigger channel box ('Channel A' -> 'A')
        channel = self.trigger_channel_box.currentText()[-1]
        self._trigger_channel = channel
        self.scope.set_trigger(channel, self._polarity_sign * self._threshold,
                               edge, is_enabled=self._is_trigger_enabled)
        self.scope.stop()

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

        self.scope.stop()

    def _calculate_num_samples(self):
        dt = self.scope.get_interval_from_timebase(self._timebase)
        pre_samples = floor(self._pre_trigger_window / dt)
        post_samples = floor(self._post_trigger_window / dt) + 1
        return pre_samples, post_samples

    def _update_run_time_label(self):
        run_time = round(self._t_prev_run_time
                         + time.time() - self._t_start_run)
        self.run_time_label.setText(f"{run_time} s")
        self.num_events_label.setText(f"({self.num_events} events)")
        # Force repaint for fast response on user input
        self.run_time_label.repaint()
        self.num_events_label.repaint()

    @QtCore.pyqtSlot()
    def fetch_data(self):
        t, [A, B] = self.scope.get_data()
        if A is not None:
            self.num_events += len(A)
            self.plot_data_signal.emit({'x': t, 'A': A, 'B': B})
        if self._is_running:
            self.start_run_signal.emit()
        if self.is_run_time_completed():
            self.stop_run()

    def is_run_time_completed(self):
        run_time = time.time() - self._t_start_run
        all_run_time = self._t_prev_run_time + run_time
        return all_run_time >= self.run_duration_box.value()

    @QtCore.pyqtSlot()
    def clear_run(self):
        self._t_prev_run_time = 0
        self._t_start_run = time.time()
        self.num_events = 0
        self._pulseheights = {'A': [], 'B': []}
        self._run_min_baseline, self._run_max_baseline = [0, 0], [0, 0]
        self._update_run_time_label()
        self.init_spectrum_plot()

    @QtCore.pyqtSlot(dict)
    def plot_data(self, data):
        x, A, B = data['x'], data['A'], data['B']

        baselines, pulseheights = [], []
        for data in A, B:
            data *= self._polarity_sign
            num_samples = int(self._pre_samples * .8)
            if self._is_baseline_correction_enabled and num_samples > 0:
                bl = data[:, :num_samples].mean(axis=1)
            else:
                bl = np.zeros(len(A))
            ph = (data.max(axis=1) - bl) * 1e3
            baselines.append(bl)
            pulseheights.append(ph)

        baselines = np.array(baselines)
        pulseheights = np.array(pulseheights)

        if self._is_upper_threshold_enabled:
            channel_idx = ['A', 'B'].index(self._trigger_channel)
            condition = (pulseheights[channel_idx, :]
                         <= self._upper_threshold * 1e3)
            A = A.compress(condition, axis=0)
            B = B.compress(condition, axis=0)
            baselines = baselines.compress(condition, axis=1)
            pulseheights = pulseheights.compress(condition, axis=1)

        for channel, values in zip(['A', 'B'], pulseheights):
            self._pulseheights[channel].extend(values)

        if len(A) > 0:
            # store min and max baselines for current run
            # axis dark magic (check carefully)
            self._run_min_baseline = np.min(
                [self._run_min_baseline, baselines.min(axis=1)], axis=0)
            self._run_max_baseline = np.max(
                [self._run_max_baseline, baselines.max(axis=1)], axis=0)

            self.update_event_plot(x, A[-1], B[-1], pulseheights[:, -1],
                                   baselines[:, -1])
            self.update_spectrum_plot()

    def init_event_plot(self):
        self.event_plot.clear()
        self.event_plot.setLabels(title='Scintillator event',
                                  bottom='Time [us]', left='Signal [V]')

    @QtCore.pyqtSlot()
    def reset_event_axes(self):
        self.event_plot.enableAutoRange(axis=pg.ViewBox.XAxis)
        self.event_plot.setYRange(-self._range - self._offset,
                                  self._range - self._offset)

    def update_event_plot(self, x, A, B, pulseheights, baselines):
        self.event_plot.clear()
        if self.ch_A_enabled_box.isChecked():
            self.event_plot.plot(x * 1e6, A,
                                 pen={'color': 'k', 'width': 2.})
        if self.ch_B_enabled_box.isChecked():
            self.event_plot.plot(x * 1e6, B,
                                 pen={'color': 'b', 'width': 2.})

        self.draw_event_plot_guides(x, baselines, pulseheights)

    def draw_event_plot_guides(self, x, baselines, pulseheights):
        phA, phB = pulseheights
        blA, blB = baselines
        plot = self.event_plot

        # mark baselines and pulseheights
        if self.ch_A_enabled_box.isChecked():
            self.draw_guide(plot, blA, 'blue')
            self.draw_guide(plot, phA / 1e3, 'purple')
        if self.ch_B_enabled_box.isChecked():
            self.draw_guide(plot, blB, 'blue')
            self.draw_guide(plot, phB / 1e3, 'purple')

        # mark trigger instant
        try:
            # right after updating settings, pre_samples may exceed old event
            self.draw_guide(plot, x[self._pre_samples] * 1e6, 'green',
                            'vertical')
        except IndexError:
            pass

        # mark trigger thresholds
        self.draw_guide(plot, self._threshold, 'green')
        if self._is_upper_threshold_enabled:
            self.draw_guide(plot, self._upper_threshold, 'green')

    def draw_guide(self, plot, pos, color, orientation='horizontal'):
        if orientation == 'vertical':
            angle = 90
        else:
            angle = 0
        color = GUIDE_COLORS[color]
        plot.addItem(pg.InfiniteLine(
            pos=pos, angle=angle,
            pen={'color': color, 'width': 2.}))

    def init_spectrum_plot(self):
        self.spectrum_plot.clear()
        self.spectrum_plot.setLabels(title='Spectrum',
                                     bottom='Pulseheight [mV]', left='Counts')

    @QtCore.pyqtSlot()
    def reset_spectrum_axes(self):
        self.spectrum_plot.enableAutoRange()

    def update_spectrum_plot(self):
        self.spectrum_plot.clear()
        x, bins, channel_counts = self.make_spectrum()
        for counts, color in zip(channel_counts, ['k', 'b']):
            if counts is not None:
                self.spectrum_plot.plot(x, counts, pen={'color': color,
                                                        'width': 2.})
        self.draw_spectrum_plot_guides()

    def make_spectrum(self):
        xrange = 2 * self._range * 1e3
        xmin = .01 * self.lld_box.value() * xrange
        xmax = .01 * self.uld_box.value() * xrange
        if xmax < xmin:
            xmax = xmin
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

    def draw_spectrum_plot_guides(self):
        min_blA, min_blB = self._run_min_baseline
        max_blA, max_blB = self._run_max_baseline
        plot = self.spectrum_plot

        self.draw_guide(plot, self._threshold * 1e3, 'green', 'vertical')
        clip_level = (self._range - self._offset)
        self.draw_guide(plot, clip_level * 1e3, 'red', 'vertical')
        if self._is_upper_threshold_enabled:
            self.draw_guide(plot, self._upper_threshold * 1e3, 'green',
                            'vertical')

        if self._is_baseline_correction_enabled:
            if self.ch_A_enabled_box.isChecked():
                lower_bound = self._threshold - min_blA
                self.draw_guide(plot, lower_bound * 1e3, 'purple', 'vertical')
                clip_level = (self._range - self._offset) - max_blA
                self.draw_guide(plot, clip_level * 1e3, 'purple', 'vertical')

                if (self._is_upper_threshold_enabled
                        and self._trigger_channel == 'A'):
                    upper_bound = self._upper_threshold - max_blA
                    self.draw_guide(plot, upper_bound * 1e3, 'purple',
                                    'vertical')

            if self.ch_B_enabled_box.isChecked():
                lower_bound = self._threshold - min_blB
                self.draw_guide(plot, lower_bound * 1e3, 'purple', 'vertical')
                clip_level = (self._range - self._offset) - max_blB
                self.draw_guide(plot, clip_level * 1e3, 'purple', 'vertical')

                if (self._is_upper_threshold_enabled
                        and self._trigger_channel == 'B'):
                    upper_bound = self._upper_threshold - max_blB
                    self.draw_guide(plot, upper_bound * 1e3, 'purple',
                                    'vertical')

    def export_spectrum_dialog(self):
        """Dialog for exporting a data file."""

        file_path, _ = QtWidgets.QFileDialog.getSaveFileName(
            self, caption="Save spectrum", directory="spectrum.csv")
        if not file_path:
            # Cancel was pressed, no file was selected
            return

        x, _, channel_counts = self.make_spectrum()
        channel_counts = [u if u is not None else [0] * len(x) for
                          u in channel_counts]

        with open(file_path, 'w') as f:
            writer = csv.writer(f)
            writer.writerow(('pulseheight', 'counts_ch_A', 'counts_ch_B'))
            for row in zip(x, *channel_counts):
                writer.writerow(row)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--fake', action='store_true',
                        help="Use fake hardware")
    args = parser.parse_args()

    qtapp = QtWidgets.QApplication(sys.argv)
    ui = UserInterface(use_fake=args.fake)
    sys.exit(qtapp.exec_())


if __name__ == '__main__':
    main()
