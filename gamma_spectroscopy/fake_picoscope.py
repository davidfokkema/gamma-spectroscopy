"""Interface to a fake 5000 Series PicoScope.

You can use this to test or demonstrate an application.

Classes
-------
FakePicoScope
    Interface to a fake 5000 Series PicoScope.
"""

import ctypes
from threading import Event, Timer
import time

import numpy as np

from .picoscope_5000a import callback_factory


class FakePicoScope:
    """Interface to a fake 5000 Series PicoScope.

    This class offers a python-friendly interface to a fake 5000 Series
    PicoScope (e.g. a PicoScope 5242D).

    Methods
    -------
    open()
        open the device
    close()
        close the device
    set_channel()
        Set up input channels
    measure()
        Start a data collection run and return the data
    measure_adc_values()
        Start a data collection run and return the data in ADC values
    set_up_buffers()
        Set up memory buffers for reading data from device
    get_adc_data()
        Return all captured data, in ADC values
    get_data()
        Return all captured data, in physical units
    get_interval_from_timebase()
        Get sampling interval for given timebase
    start_run()
        Start a run in (rapid) block mode
    wait_for_data()
        Wait for device to finish data capture
    stop()
        Stop data capture
    set_trigger()
        Set the oscilloscope trigger condition

    """
    def __init__(self, serial=None, resolution_bits=12):
        """Instantiate the class and open the device."""
        self._channels_enabled = {'A': True, 'B': True}
        self._input_voltage_ranges = {}
        self._input_offsets = {}
        self._input_adc_ranges = {}
        self._buffers = {}
        self.data_is_ready = Event()
        self._default_callback = callback_factory(self.data_is_ready)
        self._timer = None
        self._is_trigger_enabled = False

    def open(self, serial=None, resolution_bits=12):
        """Open the device.

        :param serial: (optional) Serial number of the device
        :param resolution_bits: vertical resolution in number of bits
        """
        self.set_channel('A', is_enabled=False)
        self.set_channel('B', is_enabled=False)

    def close(self):
        """Close the device."""
        pass

    def set_channel(self, channel_name, coupling_type='DC', range_value=1,
                    offset=0, is_enabled=True):
        """Set up input channels.

        :param channel_name: channel name ('A', 'B', etc.)
        :param coupling_type: 'AC' or 'DC' coupling
        :param range_value: (float) input voltage range in volts
        :param offset: analogue offset of the input signal
        :param is_enabled: enable or disable the channel
        :type is_enabled: boolean

        The input voltage range can be 10, 20, 50 mV, 100, 200, 500 mV, 1, 2,
        5 V or 10, 20, 50 V, but is given in volts. For example, a range of
        20 mV is given as 0.02.
        """
        self._range = range_value
        self._offset = offset

    def measure(self, num_pre_samples, num_post_samples, timebase=4,
                num_captures=1):
        raise NotImplementedError

    def measure_adc_values(self, num_pre_samples, num_post_samples, timebase=4,
                           num_captures=1):
        raise NotImplementedError

    def set_up_buffers(self, num_samples, num_captures=1):
        """Set up memory buffers for reading data from device.

        :param num_samples: the number of required samples per capture.
        :param num_captures: the number of captures.
        """
        pass

    def get_adc_data(self):
        raise NotImplementedError

    def get_data(self):
        """Return all captured data, in physical units.

        This method returns a tuple of time values (in seconds) and the
        captured data (in Volts).

        """
        time_values = self._calculate_time_values(self._timebase,
                                                  self._num_samples)

        V_data = []
        for channel in self._channels_enabled:
            if self._channels_enabled[channel] is True:
                V_data.append(self._create_fake_events())
            else:
                V_data.append(None)
        return time_values, V_data

    def _calculate_time_values(self, timebase, num_samples):
        """Calculate time values from timebase and number of samples."""
        interval = self.get_interval_from_timebase(timebase, num_samples)
        return interval * np.arange(num_samples) * 1e-9

    def get_interval_from_timebase(self, timebase, num_samples=1000):
        """Get sampling interval for given timebase.

        :param timebase: timebase setting (see programmers guide for reference)
        :param num_samples: number of samples required

        :returns: sampling interval in nanoseconds
        """
        if timebase <= 3:
            return 2 ** (timebase - 1) / 500e6 * 1e9
        else:
            return (timebase - 3) / 62.5e6 * 1e9

    def start_run(self, num_pre_samples, num_post_samples, timebase=4,
                  num_captures=1, callback=None):
        """Start a run in (rapid) block mode.

        Start a data collection run in 'rapid block mode' and collect a number
        of captures. Unlike the :method:`measure` and
        :method:`measure_adc_values`, which handle all details for you, this
        method returns immediately, while the device captures the requested
        data. Make sure that you *first* run :method:`set_up_buffers` to set up
        the device's memory buffers. You can supply a C-style callback to be
        notified when the device is ready, or call :method:`wait_for_data:.
        When the data is ready, call :method:`get_data` or
        :method:`get_adc_data`. When done measuring data, make sure to call
        :method:`stop`.

        :param num_pre_samples: number of samples before the trigger
        :param num_post_samples: number of samples after the trigger
        :param timebase: timebase setting (see programmers guide for reference)
        :param num_captures: number of captures to take

        :returns: data
        """
        if self._timer is not None and self._timer.is_alive():
            # Data capture already running
            return

        # save samples and captures for reference
        self._num_pre_samples = num_pre_samples
        self._num_post_samples = num_post_samples
        self._num_samples = num_pre_samples + num_post_samples
        self._timebase = timebase
        self._num_captures = num_captures

        if callback is None:
            callback = self._default_callback
        self.data_is_ready.clear()
        self._callback = callback

        if self._is_trigger_enabled:
            wait_time = np.random.exponential(scale=1 / 100.,
                                              size=num_captures).sum()
        else:
            wait_time = .001 * num_captures

        self._timer = Timer(wait_time, callback,
                            (ctypes.c_int16(), ctypes.c_int(),
                             ctypes.c_void_p()))
        self._timer.start()

    def wait_for_data(self):
        """Wait for device to finish data capture."""
        self.data_is_ready.wait()

    def stop(self):
        """Stop data capture."""
        if self._timer is not None:
            self._timer.cancel()
            self._callback(ctypes.c_int16(), ctypes.c_int(), ctypes.c_void_p())
            pass

    def set_trigger(self, channel_name, threshold=0., direction='RISING',
                    is_enabled=True, delay=0, auto_trigger=0):
        """Set the oscilloscope trigger condition.

        :param channel_name: the source channel for the trigger (e.g. 'A')
        :param threshold: the trigger threshold (in V)
        :param direction: the direction in which the signal must move to cause
            a trigger
        :param is_enabled: (boolean) enable or disable the trigger
        :param delay: the delay between the trigger occuring and the start of
            capturing data, in number of sample periods
        :param auto_trigger: the maximum amount of time to wait for a trigger
            before starting capturing data in seconds

        The direction parameter can take values of 'ABOVE', 'BELOW', 'RISING',
        'FALLING' or 'RISING_OR_FALLING'.
        """
        self._is_trigger_enabled = is_enabled
        self._trigger_threshold = threshold
        self._trigger_direction = direction

    def _create_fake_events(self):
        events = []
        for _ in range(self._num_captures):
            event = self._create_fake_event()
            events.append(event)
        return np.array(events)

    def _create_fake_event(self):
        interval = self.get_interval_from_timebase(self._timebase)
        t = interval * 1e-9 * np.arange(self._num_samples)

        if self._is_trigger_enabled:
            offset = self._num_pre_samples
            pulseheight = self._fake_pulseheight_from_spectrum()
        else:
            if self._num_samples > 1:
                offset = np.random.randint(0, self._num_samples - 1)
            else:
                offset = 0
            if np.random.random() < .1:
                pulseheight = self._fake_pulseheight_from_spectrum()
            else:
                pulseheight = 0.

        signal = -pulseheight * np.exp(-60e3 * t)
        baseline = np.random.normal(scale=10e-3, loc=5e-3)
        noise = np.random.normal(size=t.shape, scale=3e-3, loc=baseline)

        event = noise
        if offset == 0:
            event += signal
        else:
            event[offset:] += signal[:-offset]

        amin = -self._range - self._offset
        amax = self._range - self._offset
        event = event.clip(amin, amax)

        return event

    def _fake_pulseheight_from_spectrum(self):
        if np.random.random() <= .25:
            pulseheight = np.random.normal(1274e-3, scale=75e-3)
        else:
            pulseheight = np.random.normal(511e-3, scale=30e-3)
        return pulseheight
