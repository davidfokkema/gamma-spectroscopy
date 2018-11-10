from importlib import reload

from daq import picoscope_5000a

if 'dev' in globals():
    dev.close()
    reload(picoscope_5000a)

dev = picoscope_5000a.PicoScope5000A()
dev.set_channel('A', 'DC', .1)
data = dev.run_block(1000, 1000, num_captures=10)
