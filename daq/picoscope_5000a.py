"""Interface to a 5000 Series PicoScope.

Classes
-------
PicoScope5000A
    Interface to a 5000 Series PicoScope.
"""

import ctypes

import numpy as np

from picosdk.ps5000a import ps5000a as ps
from picosdk.functions import assert_pico_ok


INPUT_RANGES = {
    0.01: '10MV',
    0.02: '20MV',
    0.05: '50MV',
    0.1: '100MV',
    0.2: '200MV',
    0.5: '500MV',
    1: '1V',
    2: '2V',
    5: '5V',
    10: '10V',
    20: '20V',
    50: '50V',
}


class InvalidParameterError(Exception):
    """Error because of an invalid parameter."""

    pass


class PicoScope5000A:
    """Interface to a 5000 Series PicoScope.

    This class encapsulates the low-level PicoSDK and offers a python-friendly
    interface to a 5000 Series PicoScope (e.g. a PicoScope 5242D).

    Methods
    -------
    open()
        open the device
    close()
        close the device
    set_channel()
        Set up input channels
    run_block()
        Start a data collection run and return the data
    get_interval_from_timebase()
        Get sampling interval for given timebase

    """

    _handle = None

    def __init__(self, serial=None, resolution_bits=12):
        """Instantiate the class and open the device."""
        self._input_voltage_ranges = {}
        self._input_adc_ranges = {}
        self.open(serial, resolution_bits)

    def __del__(self):
        """Instance destructor, close device."""
        self.close()

    def open(self, serial=None, resolution_bits=12):
        """Open the device.

        :param serial: (optional) Serial number of the device
        :param resolution_bits: vertical resolution in number of bits
        """
        handle = ctypes.c_int16()
        resolution = _get_resolution_from_bits(resolution_bits)
        assert_pico_ok(ps.ps5000aOpenUnit(ctypes.byref(handle), serial,
                                          resolution))

        self._handle = handle

    def close(self):
        """Close the device."""
        assert_pico_ok(ps.ps5000aCloseUnit(self._handle))
        self._handle = None

    def set_channel(self, channel_name, coupling_type, range_value, offset=0,
                    is_enabled=True):
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
        channel = _get_channel_from_name(channel_name)
        coupling_type = _get_coupling_type_from_name(coupling_type)
        range = _get_range_from_value(range_value)
        assert_pico_ok(ps.ps5000aSetChannel(self._handle, channel, is_enabled,
                                            coupling_type, range, offset))

        self._input_voltage_ranges[channel_name] = float(range_value)
        max_adc_value = ctypes.c_int16()
        assert_pico_ok(ps.ps5000aMaximumValue(self._handle,
                                              ctypes.byref(max_adc_value)))
        self._input_adc_ranges[channel_name] = max_adc_value.value

    def run_block(self, num_pre_samples, num_post_samples, timebase=4,
                  num_captures=1):
        """Start a data collection run and return the data.

        WIP: this method only collects data on channel A.

        Start a data collection run and collect a number of captures. The data
        is returned as a twodimensional NumPy array (i.e. a 'list' of captures)
        with data values in volts. An array of time values in seconds is also
        returned.

        :param num_pre_samples: number of samples before the trigger
        :param num_post_samples: number of samples after the trigger
        :param timebase: timebase setting (see programmers guide for reference)
        :param num_captures: number of captures to take

        :returns: time_values, data
        """
        data = self._run_block_raw(num_pre_samples, num_post_samples, timebase,
                                   num_captures)

        num_samples = num_pre_samples + num_post_samples
        time_values = self._calculate_time_values(timebase, num_samples)
        data = self._rescale_adc_to_V(data)

        return time_values, data

    def _run_block_raw(self, num_pre_samples, num_post_samples, timebase,
                       num_captures):
        """Actually perform a data collection run and return raw data."""
        data = []
        num_samples = num_pre_samples + num_post_samples
        self._set_data_buffer('A', num_samples)
        for _ in range(num_captures):
            self._ps_run_block(num_pre_samples, num_post_samples, timebase)
            self._wait_for_data()
            self._get_values(num_samples)
            data.append(np.array(self._buffer))
        self._stop()
        return np.array(data)

    def _calculate_time_values(self, timebase, num_samples):
        """Calculate time values from timebase and number of samples."""
        interval = self.get_interval_from_timebase(timebase, num_samples)
        return interval * np.arange(num_samples) * 1e-9

    def _rescale_adc_to_V(self, data):
        """Rescale the ADC data and return float values in volts.

        WIP: this method only rescales correctly for channel A.
        """
        voltage_range = self._input_voltage_ranges['A']
        max_adc_value = self._input_adc_ranges['A']
        return (voltage_range * data) / max_adc_value

    def _rescale_V_to_adc(self, data):
        """Rescale float values in volts to ADC values.

        WIP: this method only rescales correctly for channel A.
        """
        voltage_range = self._input_voltage_ranges['A']
        max_adc_value = self._input_adc_ranges['A']
        output = max_adc_value * data / voltage_range
        try:
            return output.astype(np.int16)
        except AttributeError:
            return int(output)

    def get_interval_from_timebase(self, timebase, num_samples=1000):
        """Get sampling interval for given timebase.

        :param timebase: timebase setting (see programmers guide for reference)
        :param num_samples: number of samples required

        :returns: sampling interval in nanoseconds
        """
        interval = ctypes.c_float()
        assert_pico_ok(ps.ps5000aGetTimebase2(
            self._handle, timebase, num_samples, ctypes.byref(interval), None,
            0))
        return interval

    def _set_data_buffer(self, channel, num_samples):
        """Set up data buffer.

        :param channel: channel name ('A', 'B', etc.)
        :param num_samples: number of samples required
        """
        channel = _get_channel_from_name(channel)
        self._buffer = (ctypes.c_int16 * num_samples)()
        assert_pico_ok(ps.ps5000aSetDataBuffer(
            self._handle, channel, ctypes.byref(self._buffer), num_samples, 0,
            0))

    def _ps_run_block(self, num_pre_samples, num_post_samples, timebase):
        """Run in block mode."""
        assert_pico_ok(ps.ps5000aRunBlock(
            self._handle, num_pre_samples, num_post_samples, timebase, None, 0,
            None, None))

    def _wait_for_data(self):
        """Wait for device to finish data capture."""
        ready = ctypes.c_int16(0)
        while not ready:
            assert_pico_ok(ps.ps5000aIsReady(self._handle,
                                             ctypes.byref(ready)))

    def _get_values(self, num_samples):
        """Get data from device and store in buffer."""
        num_samples = ctypes.c_int32(num_samples)
        overflow = ctypes.c_int16()
        assert_pico_ok(ps.ps5000aGetValues(
            self._handle, 0, ctypes.byref(num_samples), 0, 0, 0,
            ctypes.byref(overflow)))

    def _stop(self):
        """Stop data capture."""
        assert_pico_ok(ps.ps5000aStop(self._handle))

    def set_trigger(self, channel, threshold, direction, is_enabled=True,
                    delay=0, auto_trigger=0):
        """Set the oscilloscope trigger condition.

        :param channel: the source channel for the trigger (e.g. 'A')
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
        channel = _get_channel_from_name(channel)
        direction = _get_trigger_direction_from_name(direction)
        assert_pico_ok(ps.ps5000aSetSimpleTrigger(
            self._handle, is_enabled, channel, threshold, direction, delay,
            auto_trigger))


def _get_resolution_from_bits(resolution_bits):
    """Return the resolution from the number of bits."""
    if resolution_bits in [8, 12, 14, 15, 16]:
        def_name = f"PS5000A_DR_{resolution_bits}BIT"
    else:
        raise InvalidParameterError(f"A resolution of {resolution_bits}-bits "
                                    "is not supported")
    return ps.PS5000A_DEVICE_RESOLUTION[def_name]


def _get_channel_from_name(channel_name):
    """Return the channel from the channel name."""
    if channel_name in ['A', 'B', 'C', 'D']:
        def_name = f"PS5000A_CHANNEL_{channel_name}"
    else:
        raise InvalidParameterError(f"Channel {channel_name} is not supported")
    return ps.PS5000A_CHANNEL[def_name]


def _get_coupling_type_from_name(coupling_type_name):
    """Return the coupling type from the coupling type name."""
    if coupling_type_name in ['AC', 'DC']:
        def_name = f"PS5000A_{coupling_type_name}"
    else:
        raise InvalidParameterError(f"Coupling type {coupling_type_name} is "
                                    "not supported")
    return ps.PS5000A_COUPLING[def_name]


def _get_range_from_value(range):
    """Return the range from the range in volts."""
    if range in INPUT_RANGES:
        range_name = INPUT_RANGES[range]
        def_name = f"PS5000A_{range_name}"
    else:
        raise InvalidParameterError(f"Range {range} V is not supported")
    return ps.PS5000A_RANGE[def_name]


def _get_trigger_direction_from_name(direction_name):
    """Return the trigger direction from the direction name."""
    if direction_name in ['ABOVE', 'BELOW', 'RISING', 'FALLING',
                          'RISING_OR_FALLING']:
        def_name = f"PS5000A_{direction_name}"
    else:
        raise InvalidParameterError(f"Trigger direction {direction_name} is "
                                    "not supported")
    return ps.PS5000A_THRESHOLD_DIRECTION[def_name]
