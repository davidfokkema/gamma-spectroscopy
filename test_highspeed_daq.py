import time

from daq import picoscope_5000a


dev = picoscope_5000a.PicoScope5000A(resolution_bits=12)
dev.set_channel('A', 'DC', .5, offset=.4)
dev.set_trigger('A', 0.03, 'FALLING')

NUM_CAPTURES = 100
N = 0
t0 = time.time()
while time.time() - t0 < 5:
    dev.measure_adc_values(100, 100, 2, num_captures=NUM_CAPTURES)
    N += NUM_CAPTURES
t1 = time.time()

print(f"Event rate was {N / (t1 - t0):.1f} Hz.")
