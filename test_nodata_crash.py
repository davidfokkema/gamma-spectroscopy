import ctypes
import time

from daq.picoscope_5000a import PicoScope5000A
from picosdk.constants import PICO_STATUS_LOOKUP


@ctypes.CFUNCTYPE(None, ctypes.c_int16, ctypes.c_int, ctypes.c_void_p)
def my_callback(handle, status, parameters):
    print("CALLBACK")
    print(f"{PICO_STATUS_LOOKUP[status]}")
    dev.get_adc_data()


dev = PicoScope5000A()
dev.set_channel('A', 'DC', 1)
# dev.set_trigger('A', 1, 'RISING')
dev.set_up_buffers(200, 10000)
dev.start_run(100, 100, 20, num_captures=10000, callback=my_callback)
# time.sleep(.5)
print("ST", end=None)
dev.stop()
time.sleep(.1)
print("OP")
time.sleep(1)
