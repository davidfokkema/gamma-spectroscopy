from importlib import reload
import matplotlib.pyplot as plt

from daq import picoscope_5000a

if 'dev' in globals():
    dev.close()
    reload(picoscope_5000a)

dev = picoscope_5000a.PicoScope5000A()
dev.set_channel('A', 'DC', 10, offset=1)
t, data = dev.run_block(1000, 1000, timebase=2000, num_captures=10)

plt.figure()
plt.plot(t * 1e3, data.T)
plt.xlabel('Time [ms]')
plt.show()