import ctypes
import time
from threading import Event

from picosdk.ps5000a import ps5000a as ps
from picosdk.functions import assert_pico_ok

from daq import picoscope_5000a


data_is_ready = Event()


@ctypes.CFUNCTYPE(None, ctypes.c_int16, ctypes.c_int, ctypes.c_void_p)
def my_callback(handle, status, parameters):
    data_is_ready.set()


dev = picoscope_5000a.PicoScope5000A()
dev.set_channel('A', 'DC', 10, offset=1)

t0 = time.time()
for i in range(100):
    assert_pico_ok(ps.ps5000aRunBlock(
        dev._handle, 1000, 1000, 2000, None, 0, None, None))
    ready = ctypes.c_int16(0)
    while not ready:
        assert_pico_ok(ps.ps5000aIsReady(dev._handle,
                                         ctypes.byref(ready)))
t1 = time.time()
print(f"Run without callbacks took {t1 - t0}.")

t0 = time.time()
for i in range(100):
    assert_pico_ok(ps.ps5000aRunBlock(
        dev._handle, 1000, 1000, 2000, None, 0, my_callback, None))
    data_is_ready.wait()
    data_is_ready.clear()
t1 = time.time()
print(f"Run with callbacks took {t1 - t0}.")
