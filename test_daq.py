from importlib import reload
from matplotlib.pylab import *

from daq import picoscope_5000a

if 'dev' in globals():
    dev.close()
    reload(picoscope_5000a)

dev = picoscope_5000a.PicoScope5000A()
dev.set_channel('A', 'DC', 10, offset=1)
t, data = dev.run_block(1000, 1000, timebase=2000, num_captures=10)

figure()
plot(t * 1e3, data.T)
xlabel('Time [ms]')
show()
