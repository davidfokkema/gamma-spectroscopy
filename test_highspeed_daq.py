import time

from daq import picoscope_5000a


dev = picoscope_5000a.PicoScope5000A()
dev.set_channel('A', 'DC', .5)
# dev.set_trigger('A', 0.04, 'FALLING')

NUM_CAPTURES = 10000
N = 0
t0 = time.time()
while time.time() - t0 < 5:
    dev.measure_adc_values(1000, 1000, 2, num_captures=NUM_CAPTURES)
    N += NUM_CAPTURES
t1 = time.time()

print(f"Event rate was {N / (t1 - t0):.1f} Hz.")
