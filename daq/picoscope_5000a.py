"""Interface to a 5000 Series PicoScope.

Classes
-------
PicoScope5000A
    Interface to a 5000 Series PicoScope.
"""

import ctypes

from picosdk.ps5000a import ps5000a as ps
from picosdk.functions import assert_pico_ok


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


def _get_resolution_from_bits(resolution_bits):
    """Return the resolution from the number of bits."""

    if resolution_bits in [8, 12, 14, 15, 16]:
        res_name = f"PS5000A_DR_{resolution_bits}BIT"
    else:
        raise InvalidParameterError(f"A resolution of {resolution_bits}-bits "
                                    "is not supported")
    return ps.PS5000A_DEVICE_RESOLUTION[res_name]
