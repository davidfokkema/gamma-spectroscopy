from importlib import reload
import matplotlib.pyplot as plt

import numpy as np

from daq import picoscope_5000a

if 'dev' in globals():
    dev.close()
    reload(picoscope_5000a)

dev = picoscope_5000a.PicoScope5000A()
dev.set_channel('A', 'DC', 10, offset=0)
dev.set_trigger('A', 100, 'RISING_OR_FALLING')

print(dev._rescale_V_to_adc(.4))
print(dev._rescale_V_to_adc(np.array([.4, .5, .6])))

print(dev._rescale_adc_to_V(10))
print(dev._rescale_adc_to_V(np.array([10, 20, 1000])))

# t, data = dev.run_block(0, 1000, timebase=2000, num_captures=10)

plt.figure()
plt.plot(t * 1e3, data.T)
plt.xlabel('Time [ms]')
plt.show()
