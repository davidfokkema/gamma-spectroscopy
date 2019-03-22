from daq import picoscope_5000a

dev = picoscope_5000a.PicoScope5000A()
dev.set_channel('A', 'DC', 1.)
dev.set_channel('B', 'DC', 1.)
dev.set_trigger('A', is_enabled=False)

# data = dev.measure(1, 1)
# print(data)

dev.set_up_buffers(2, num_captures=10)
dev.start_run(1, 1, num_captures=10)
dev.wait_for_data()
data = dev.get_data()
print(data)
