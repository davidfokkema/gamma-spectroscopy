import argparse
import csv
import ctypes
from math import floor
import sys
from pathlib import Path
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

# Custom symbol for use in (mostly) histogram plot
histogram_symbol = pg.QtGui.QPainterPath()
histogram_symbol.moveTo(0, -.5)
histogram_symbol.lineTo(0, .5)

PLOT_OPTIONS = {
    'lines': {'A': {'pen': {'color': 'w', 'width': 2.}},
              'B': {'pen': {'color': (255, 200, 0), 'width': 2.}},
             },
    'marks': {'A': {'pen': None, 'symbol': histogram_symbol,
                    'symbolPen': 'w', 'symbolSize': 2},
              'B': {'pen': None, 'symbol': histogram_symbol,
                    'symbolPen': (255, 200, 0), 'symbolSize': 2},
             },
}

def create_callback(signal):
    @ctypes.CFUNCTYPE(None, ctypes.c_int16, ctypes.c_int, ctypes.c_void_p)
    def my_callback(handle, status, parameters):
        signal.emit()
    return my_callback


class UserInterface(QtWidgets.QMainWindow):

    POLARITY = ['Positive', 'Negative']
    POLARITY_SIGN = [1, -1]

    COUPLING = ['AC', 'DC']

    start_run_signal = QtCore.pyqtSignal()
    new_data_signal = QtCore.pyqtSignal()
    plot_data_signal = QtCore.pyqtSignal(dict)

    run_timer = QtCore.QTimer(interval=1000)

    num_events = 0

    _is_running = False
    _trigger_channel = 'A'
    _coupling = 'AC'
    _is_trigger_enabled = False
    _is_upper_threshold_enabled = False
    _pulse_polarity = 'Positive'
    _polarity_sign = 1
    _is_baseline_correction_enabled = True
    _show_guides = False
    _show_marks = False
    _plot_options = PLOT_OPTIONS['lines']

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
    _run_time = 0
    _t_prev_run_time = 0

    _write_output = False
    _output_path = Path.home() / 'Documents'
    _run_number = 0
    _output_filename = None
    _output_file = None

    def __init__(self, use_fake=False):
        super().__init__()

        self._pulseheights = {'A': [], 'B': []}
        self._baselines = {'A': [], 'B': []}

        if use_fake:
            self.scope = FakePicoScope()
        else:
            self.scope = PicoScope5000A()

        self.init_ui()

    def closeEvent(self, event):
        self._is_running = False
        self.scope.stop()

    def init_ui(self):
        pg.setConfigOption('background', 'k')
        pg.setConfigOption('foreground', 'w')
        pg.setConfigOption('antialias', True)

        ui_path = resource_filename('gamma_spectroscopy',
                                    'gamma_spectroscopy_gui.ui')
        layout = uic.loadUi(ui_path, self)

        # Menubar
        menubar = QtWidgets.QMenuBar()

        export_spectrum_action = QtWidgets.QAction('&Export spectrum', self)
        export_spectrum_action.setShortcut('Ctrl+S')
        export_spectrum_action.triggered.connect(self.export_spectrum_dialog)

        write_output_action = QtWidgets.QAction('&Write output files', self)
        write_output_action.setShortcut('Ctrl+O')
        write_output_action.triggered.connect(self.write_output_dialog)

        file_menu = menubar.addMenu('&File')
        file_menu.addAction(export_spectrum_action)
        file_menu.addAction(write_output_action)

        layout.setMenuBar(menubar)

        statusbar = QtWidgets.QStatusBar()
        self.label_status = QtWidgets.QLabel("")
        statusbar.addWidget(self.label_status)

        layout.setStatusBar(statusbar)

        self.start_run_signal.connect(self.start_scope_run)
        self.start_run_signal.connect(self._update_run_label)

        self.new_data_signal.connect(self.fetch_data)
        self.callback = create_callback(self.new_data_signal)

        self.plot_data_signal.connect(self.plot_data)

        self.range_box.addItems(INPUT_RANGES.values())
        self.range_box.currentIndexChanged.connect(self.set_range)
        self.range_box.setCurrentIndex(5)
        self.polarity_box.addItems(self.POLARITY)
        self.polarity_box.currentIndexChanged.connect(self.set_polarity)
        self._pulse_polarity = self.POLARITY[0]
        self.coupling_box.addItems(self.COUPLING)
        self.coupling_box.currentIndexChanged.connect(self.set_coupling)
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
        self.toggle_guides_button1.clicked.connect(self.toggle_guides)
        self.toggle_guides_button2.clicked.connect(self.toggle_guides)
        self.toggle_markslines_button1.clicked.connect(
            self.toggle_show_marks_or_lines)
        self.toggle_markslines_button2.clicked.connect(
            self.toggle_show_marks_or_lines)

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
        if not self.is_run_time_completed():
            self._is_running = True
            self._t_start_run = time.time()
            self._run_time = 0
            self._update_run_time_label()
            self.run_timer.start()
            self.start_run_signal.emit()
            self.run_stop_button.setText("Stop")
            self.single_button.setDisabled(True)
            self._update_run_label()
            if self._write_output:
                self.open_output_file()
                writer = csv.writer(self._output_file)
                writer.writerow(('time_A','pulse_height_A',
                                 'time_B','pulse_height_B'))

    def stop_run(self):
        self._is_running = False
        self._update_run_time_label()
        self.scope.stop()
        self.run_timer.stop()
        self._run_time = time.time() - self._t_start_run
        self._t_prev_run_time += self._run_time
        self.run_stop_button.setText("Run")
        self.single_button.setDisabled(False)
        if self._write_output:
            self.write_info_file()
            self.close_output_file()

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
        if not self._trigger_channel == 'A OR B':
            self._is_upper_threshold_enabled = state
            self.scope.stop()

    @QtCore.pyqtSlot(int)
    def set_polarity(self, idx):
        self._pulse_polarity = self.POLARITY[idx]
        self._polarity_sign = self.POLARITY_SIGN[idx]
        self.set_channel()
        self.set_trigger()

    @QtCore.pyqtSlot(int)
    def set_coupling(self, idx):
        self._coupling = self.COUPLING[idx]
        self.set_channel()

    @QtCore.pyqtSlot(int)
    def set_baseline_correction_state(self, state):
        self._is_baseline_correction_enabled = state

    def set_channel(self):
        self._offset = np.interp(self._offset_level, [-100, 100],
                                 [-self._range, self._range])
        self.scope.set_channel('A', self._coupling, self._range,
                               self._polarity_sign * self._offset)
        self.scope.set_channel('B', self._coupling, self._range,
                               self._polarity_sign * self._offset)
        self.event_plot.setYRange(-self._range - self._offset,
                                  self._range - self._offset)
        self.scope.stop()

    def set_trigger(self):
        edge = 'RISING' if self._pulse_polarity == 'Positive' else 'FALLING'
        if self.trigger_channel_box.currentText() == 'A OR B':
            self._trigger_channel = 'A OR B'
            self.scope.set_trigger_A_OR_B(self._polarity_sign * self._threshold,
                                          edge,
                                          is_enabled=self._is_trigger_enabled)
            self._upper_trigger_state = False
            self.upper_trigger_box.setCheckable(False)
        else:
            # get last letter of trigger channel box ('Channel A' -> 'A')
            channel = self.trigger_channel_box.currentText()[-1]
            self._trigger_channel = channel
            self.scope.set_trigger(channel,
                                   self._polarity_sign * self._threshold,
                                   edge, is_enabled=self._is_trigger_enabled)
            self.upper_trigger_box.setCheckable(True)
        if self._show_guides:
            self.draw_spectrum_plot_guides()
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

    def _update_run_label(self):
        self.run_number_label.setText(f"{self._run_number}")
        self.run_number_label.repaint()

    def _update_status_bar(self):
        if self._write_output:
            status_message = f'Output directory: {str(self._output_path)}'
        else:
            status_message = ''
        self.label_status.setText(status_message)

    @QtCore.pyqtSlot()
    def toggle_guides(self):
        self._show_guides = not self._show_guides

    @QtCore.pyqtSlot()
    def toggle_show_marks_or_lines(self):
        self._show_marks = not self._show_marks
        if self._show_marks:
            self._plot_options = PLOT_OPTIONS['marks']
        else:
            self._plot_options = PLOT_OPTIONS['lines']

    @QtCore.pyqtSlot()
    def fetch_data(self):
        t, [A, B] = self.scope.get_data()
        if A is not None:
            self.num_events += len(A)
            self.plot_data_signal.emit({'x': t, 'A': A, 'B': B})
        if self._is_running:
            if self.is_run_time_completed():
                self.stop_run()
            else:
                self.start_run_signal.emit()

    def is_run_time_completed(self):
        run_time = self._t_prev_run_time
        if self._is_running:
            run_time += time.time() - self._t_start_run
        return run_time >= self.run_duration_box.value()

    @QtCore.pyqtSlot()
    def clear_run(self):
        self._t_prev_run_time = 0
        self._t_start_run = time.time()
        self.num_events = 0
        self._pulseheights = {'A': [], 'B': []}
        self._baselines = {'A': [], 'B': []}
        self._update_run_time_label()
        self.init_spectrum_plot()

    @QtCore.pyqtSlot(dict)
    def plot_data(self, data):
        x, A, B = data['x'], data['A'], data['B']
        times, baselines, pulseheights = [], [], []
        for data in A, B:
            data *= self._polarity_sign
            num_samples = int(self._pre_samples * .8)
            if self._is_baseline_correction_enabled and num_samples > 0:
                bl = data[:, :num_samples].mean(axis=1)
            else:
                bl = np.zeros(len(A))
            ph = (data.max(axis=1) - bl) * 1e3
            ts = x[np.argmax(data, axis=1)]
            times.append(ts)
            baselines.append(bl)
            pulseheights.append(ph)

        times = np.array(times)
        baselines = np.array(baselines)
        pulseheights = np.array(pulseheights)

        if self._write_output and not self._output_file.closed:
            writer = csv.writer(self._output_file)
            for row in zip(times[0], pulseheights[0], times[1], pulseheights[1]):
                writer.writerow(row)

        if self._is_upper_threshold_enabled:
            if not self._trigger_channel == 'A OR B':
                channel_idx = ['A', 'B'].index(self._trigger_channel)
                condition = (pulseheights[channel_idx, :]
                            <= self._upper_threshold * 1e3)
                A = A.compress(condition, axis=0)
                B = B.compress(condition, axis=0)
                baselines = baselines.compress(condition, axis=1)
                pulseheights = pulseheights.compress(condition, axis=1)

        for channel, tvalues, blvalues, phvalues \
            in zip(['A', 'B'], times, baselines, pulseheights):
            self._baselines[channel].extend(blvalues)
            self._pulseheights[channel].extend(phvalues)

        if len(A) > 0:
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
            self.event_plot.plot(x * 1e6, A, **self._plot_options['A'])
        if self.ch_B_enabled_box.isChecked():
            self.event_plot.plot(x * 1e6, B, **self._plot_options['B'])

        if self._show_guides:
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
        if self._is_trigger_enabled:
            self.draw_guide(plot, self._threshold, 'green')
        if self._is_upper_threshold_enabled \
           and not self._trigger_channel == 'A OR B':
            self.draw_guide(plot, self._upper_threshold, 'green')

    def draw_guide(self, plot, pos, color, orientation='horizontal', width=2.):
        if orientation == 'vertical':
            angle = 90
        else:
            angle = 0
        color = GUIDE_COLORS[color]
        plot.addItem(pg.InfiniteLine(
            pos=pos, angle=angle,
            pen={'color': color, 'width': width}))

    def init_spectrum_plot(self):
        self.spectrum_plot.clear()
        self.spectrum_plot.setLabels(title='Spectrum',
                                     bottom='Pulseheight [mV]', left='Counts')

    @QtCore.pyqtSlot()
    def reset_spectrum_axes(self):
        self.spectrum_plot.enableAutoRange()

    def update_spectrum_plot(self):
        if len(self._baselines['A']) > 0:
            self.spectrum_plot.clear()
            x, bins, channel_counts = self.make_spectrum()
            for counts, channel in zip(channel_counts, ['A', 'B']):
                if counts is not None:
                    self.spectrum_plot.plot(
                        x, counts, **self._plot_options[channel])
            if self._show_guides:
                self.draw_spectrum_plot_guides()

    def make_spectrum(self):
        #xrange = 2 * self._range * 1e3
        xrange = (self._range - self._offset) * 1e3
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
        min_blA = np.percentile(self._baselines['A'], 5)
        min_blB = np.percentile(self._baselines['B'], 5)
        # max_blA = np.percentile(self._baselines['A'], 95)
        # max_blB = np.percentile(self._baselines['B'], 95)
        plot = self.spectrum_plot

        if self._is_trigger_enabled:
            self.draw_guide(plot, self._threshold * 1e3, 'green', 'vertical')

        clip_level = (self._range - self._offset)
        self.draw_guide(plot, clip_level * 1e3, 'red', 'vertical')

        if self._is_upper_threshold_enabled  \
           and not self._trigger_channel == 'A OR B':
            self.draw_guide(plot, self._upper_threshold * 1e3, 'green',
                            'vertical')

        if self._is_baseline_correction_enabled:
            if self.ch_A_enabled_box.isChecked():
                lower_bound = self._threshold - min_blA
                self.draw_guide(plot, lower_bound * 1e3, 'purple', 'vertical')
                # REALLY think about these ones (only look for actual clipping?)
                # clip_level = (self._range - self._offset) - max_blA
                # self.draw_guide(plot, clip_level * 1e3, 'purple', 'vertical')

                # if (self._is_upper_threshold_enabled
                #         and self._trigger_channel == 'A'):
                #     upper_bound = self._upper_threshold - max_blA
                #     self.draw_guide(plot, upper_bound * 1e3, 'purple',
                #                     'vertical')

            if self.ch_B_enabled_box.isChecked():
                lower_bound = self._threshold - min_blB
                self.draw_guide(plot, lower_bound * 1e3, 'purple', 'vertical')
                # REALLY think about these ones (only look for actual clipping?)
                # clip_level = (self._range - self._offset) - max_blB
                # self.draw_guide(plot, clip_level * 1e3, 'purple', 'vertical')

                # if (self._is_upper_threshold_enabled
                #         and self._trigger_channel == 'B'):
                #     upper_bound = self._upper_threshold - max_blB
                #     self.draw_guide(plot, upper_bound * 1e3, 'purple',
                #                     'vertical')

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

        with open(file_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(('pulseheight', 'counts_ch_A', 'counts_ch_B'))
            for row in zip(x, *channel_counts):
                writer.writerow(row)

    def write_output_dialog(self):
        self._write_output = True
        file_path = QtWidgets.QFileDialog.getExistingDirectory(self,
            caption="Choose directory for output files",
            directory=str(self._output_path.absolute()))
        self._output_path = Path(file_path)
        self._update_status_bar()
        #print('Output directory: {}'.format(self._output_path))

    def open_output_file(self):
        for x in Path(self._output_path).glob('*.csv'):
            if x.is_file() and x.name[0:3] == 'Run':
                self._run_number = int(x.name[3:7]) + 1

        self._output_filename =  self._output_path / 'Run{0:04d}.csv'\
                                                       .format(self._run_number)

        try:
             self._output_file = open(self._output_filename, 'w',
                                      newline='')
             return 1
        except IOError:
            print('Error: Unable to open: {}'\
                  .format(self._output_filename))
            return 0

    def close_output_file(self):
        try:
            self._output_file.close()
            return 1
        except IOError:
            print('Error: Unable to close: {}'\
                  .format(self._output_filename))
            return 0

    def write_info_file(self):
        info_filename = self._output_filename.with_suffix('.info')
        try:
            info_file = open(info_filename, 'w', newline='', encoding="utf-8")
        except IOError:
            print(f'Error: Unable to open: {info_filename}\n')
        info_file.write(f'Start time: {time.ctime(self._t_start_run)}\n')
        info_file.write(f'Run time: {self._run_time:.1f} s\n')
        info_file.write(f'Coupling: {self._coupling}\n')
        info_file.write(f'Baseline correction: {self._is_baseline_correction_enabled}\n')
        if self._is_trigger_enabled:
            info_file.write(f'Trigger channel: {self._trigger_channel}\n')
            info_file.write(f'Threshold: {self._threshold:.3f} V\n')
        else:
            info_file.write('Untriggered\n')
        info_file.write('Pre-trigger window: {0:.2f} μs\n'\
                        .format(self._pre_trigger_window/1e3))
        info_file.write('Post-trigger window {0:.2f} μs\n'\
                        .format(self._post_trigger_window/1e3))
        info_file.write(f'Samples per capture: {self._num_samples}\n')
        info_file.write(f'Captures per block: {self.num_captures_box.value()}\n')
        info_file.close()

def main():
    global qtapp

    parser = argparse.ArgumentParser()
    parser.add_argument('--fake', action='store_true',
                        help="Use fake hardware")
    args = parser.parse_args()

    qtapp = QtWidgets.QApplication(sys.argv)
    ui = UserInterface(use_fake=args.fake)
    sys.exit(qtapp.exec_())


if __name__ == '__main__':
    main()
