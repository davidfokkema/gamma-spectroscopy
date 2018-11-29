import time

import tables

from daq import picoscope_5000a


resolution = 12
timebase = 2
pre_trigger_samples = 200
post_trigger_samples = 150
num_captures = 1000


class Gammas(tables.IsDescription):

    t = tables.TimeCol()
    trace = tables.FloatCol(
        shape=(pre_trigger_samples + post_trigger_samples,))


data = tables.open_file('data.h5', 'w')
table = data.create_table('/', 'events', Gammas)

dev = picoscope_5000a.PicoScope5000A(resolution_bits=resolution)
dev.set_channel('A', 'DC', 0.5, offset=0.4)
dev.set_trigger('A', .03, 'FALLING')

dt = dev.get_interval_from_timebase(timebase, pre_trigger_samples +
                                    post_trigger_samples)
print(f"Timebase is set to {dt} ns/sample.")

N = 0
t0 = time.time()
row = table.row
try:
    while time.time() - t0 < 900:
        t, traces = dev.measure(pre_trigger_samples, post_trigger_samples,
                                timebase, num_captures=num_captures)
        # peak_value = -trace.min()
        # if .234 <= peak_value < .254:
        for trace in traces:
            # row['t'] = time.time()
            row['trace'] = trace
            row.append()
            N += 1
except KeyboardInterrupt:
    dev.stop()
t1 = time.time()

data.create_array('/', 't', t)

print(f"Event rate was {N / (t1 - t0):.1f} Hz.")

data.flush()
