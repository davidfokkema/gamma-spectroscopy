import time

from daq import picoscope_5000a


dev = picoscope_5000a.PicoScope5000A()
dev.set_channel('A', 'DC', .5)
# dev.set_trigger('A', 0.04, 'FALLING')

N = 0
t0 = time.time()
while time.time() - t0 < 5:
    t, trace = dev.measure(100, 100, 2)
    N += 1
t1 = time.time()

print(f"Event rate was {N / (t1 - t0):.1f} Hz.")
