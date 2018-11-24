import time

from daq import picoscope_5000a


dev = picoscope_5000a.PicoScope5000A(resolution_bits=12)
dev.set_channel('A', 'DC', .05)
dev.set_trigger('A', 0, 'FALLING')

NUM_CAPTURES = 100
N = 0
t0 = time.time()
while time.time() - t0 < 5:
    # t, data = dev.measure(100, 100, 2, num_captures=NUM_CAPTURES)
    dev.set_up_buffers(200, NUM_CAPTURES)
    dev.start_run(100, 100, 2, NUM_CAPTURES)
    dev.wait_for_data()
    t, data = dev.get_data()
    N += NUM_CAPTURES
t1 = time.time()
dev.stop()

print(f"Event rate was {N / (t1 - t0):.1f} Hz.")
