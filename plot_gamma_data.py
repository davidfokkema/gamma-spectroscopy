import tables

import matplotlib.pylab as plt


data = tables.open_file('data.h5')
t = data.root.t.read()
traces = data.root.events.col('trace')

plt.figure()
plt.plot(t * 1e6, traces[:50,].T * 1e3)
plt.xlabel("Time [us]")
plt.ylabel("Voltage [mV]")

plt.figure()
plt.title("Uncorrected pulseheights")
ph = -traces.min(axis=1)
plt.hist(ph, bins=100, histtype='step', log=True)

plt.figure()
plt.title("Pulseheights corrected for baseline")
bl = traces[:,:5].mean(axis=1)
ph = -(traces.T - bl).min(axis=0)
plt.hist(ph, bins=100, histtype='step', log=False)

plt.figure()
plt.title("Pulseintegrals corrected for baseline (no rejection)")
bl = traces[:,:5].mean(axis=1)
pi = -(traces.T - bl).sum(axis=0)
plt.hist(pi, bins=100, histtype='step', log=True)

plt.show()
