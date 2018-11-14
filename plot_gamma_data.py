import tables

import matplotlib.pylab as plt


data = tables.open_file('data.h5')
t = data.root.t.read()
traces = data.root.events.col('trace')

plt.plot(t * 1e6, traces.T * 1e3)
plt.xlabel("Time [us]")
plt.ylabel("Voltage [mV]")
plt.show()
