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
        self.open(serial, resolution_bits)

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

    def set_channel(self, channel, coupling_type, range, offset=0,
                    is_enabled=True):
        """Set up input channels.

        :param channel: channel name ('A', 'B', etc.)
        :param coupling_type: 'AC' or 'DC' coupling
        :param range: (float) input voltage range in volts
        :param offset: analogue offset of the input signal
        :param is_enabled: enable or disable the channel
        :type is_enabled: boolean

        The input voltage range can be 10, 20, 50 mV, 100, 200, 500 mV, 1, 2,
        5 V or 10, 20, 50 V, but is given in volts. For example, a range of
        20 mV is given as 0.02.
        """
        channel = _get_channel_from_name(channel)
        coupling_type = _get_coupling_type_from_name(coupling_type)
        range = _get_range_from_value(range)
        assert_pico_ok(ps.ps5000aSetChannel(self._handle, channel, is_enabled,
                                            coupling_type, range, offset))

    def run_block(self, num_pre_samples, num_post_samples, timebase=4,
                  num_captures=1):
        """Start a data collection run and return the data.

        WIP: this method only collects data on channel A.

        Start a data collection run and collect a number of captures. The data
        is returned as a twodimensional NumPy array (i.e. a 'list' of
        captures). An array of time values is also returned.

        :param num_pre_samples: number of samples before the trigger
        :param num_post_samples: number of samples after the trigger
        :param timebase: timebase setting (see programmers guide for reference)
        :param num_captures: number of captures to take

        :returns: t, data
        """
        data = []
        num_samples = num_pre_samples + num_post_samples
        self._set_data_buffer('A', num_samples)
        for _ in range(num_captures):
            self._run_block(num_pre_samples, num_post_samples, timebase)
            self._wait_for_data()
            self._get_values(num_samples)
            data.append(np.array(self._buffer))
        self._stop()

        interval = self.get_interval_from_timebase(timebase, num_samples)
        t = interval * np.arange(num_samples)
        return t, np.array(data)

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

    def _run_block(self, num_pre_samples, num_post_samples, timebase):
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
