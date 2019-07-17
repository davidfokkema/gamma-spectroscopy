import ctypes
from threading import Timer, Event
import time


def callback_factory(event):
    """Return callback that will signal event when called."""
    @ctypes.CFUNCTYPE(None)  # , ctypes.c_int16, ctypes.c_int, ctypes.c_void_p)
    def data_is_ready_callback():  # handle, status, parameters):
        """Signal that data is ready when called by PicoSDK."""
        print("data is ready")
        event.set()
    return data_is_ready_callback


if __name__ == '__main__':
    timer_event = Event()
    callback = callback_factory(timer_event)

    print(timer_event.is_set())
    timer = Timer(1.0, callback)
    timer.start()
    print("Started timer.")

    time.sleep(2)
    print(timer_event.is_set())
