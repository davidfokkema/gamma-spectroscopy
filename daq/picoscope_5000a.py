import ctypes

from picosdk.ps5000a import ps5000a as ps
from picosdk.functions import assert_pico_ok


class InvalidParameterError(Exception):
    pass


class PicoScope5000A:

    def __init__(self, serial=None, resolution_bits=12):
        handle = ctypes.c_int16()
        resolution = self._get_resolution_from_bits(resolution_bits)
        assert_pico_ok(ps.ps5000aOpenUnit(ctypes.byref(handle), serial,
                                          resolution))

        self._handle = handle

    def close(self):
        assert_pico_ok(ps.ps5000aCloseUnit(self._handle))
        self._handle = None

    def _get_resolution_from_bits(self, resolution_bits):
        if resolution_bits in [8, 12, 14, 15, 16]:
            res_name = f"PS5000A_DR_{resolution_bits}BIT"
        else:
            raise InvalidParameterError(f"Resolution {resolution_bits}-bits "
                                         "not supported")
        return ps.PS5000A_DEVICE_RESOLUTION[res_name]
