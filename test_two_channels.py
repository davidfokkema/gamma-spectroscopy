from daq import picoscope_5000a

dev = picoscope_5000a.PicoScope5000A()
dev.set_channel('A', 'DC', 1.)
dev.set_trigger('A', is_enabled=False)
data = dev.measure(10, 10)
print(data)
