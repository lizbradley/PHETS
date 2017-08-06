import math


a_freq = 440.
c_freq = 261.6


a_big = (1 / a_freq) * math.pi
a_small = (1 / a_freq) / math.pi


c_big = (1 / c_freq) * math.pi
c_small = (1 / c_freq) / math.pi

print 'A big: {:.5f} \t A small: {:.5f}'.format(a_big, a_small)
print 'C big: {:.5f} \t C small: {:.5f}'.format(c_big, c_small)