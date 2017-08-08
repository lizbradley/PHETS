import math
from config import WAV_SAMPLE_RATE


a_freq = 440.		# idx 40
c_freq = 261.6		# idx 49


a_big = (1 / a_freq) * math.pi
a_small = (1 / a_freq) / math.pi


c_big = (1 / c_freq) * math.pi
c_small = (1 / c_freq) / math.pi

print 'A big: \t\t {:.7f} seconds \t {} samples'.format(a_big, round(a_big * WAV_SAMPLE_RATE))
print 'A small: \t {:.7f} seconds \t {} samples'.format(a_small, round(a_small * WAV_SAMPLE_RATE))
print ''
print 'C big: \t\t {:.7f} seconds \t {} samples'.format(c_big, round(c_big * WAV_SAMPLE_RATE))
print 'C small: \t {:.7f} seconds \t {} samples'.format(c_small, round(c_small * WAV_SAMPLE_RATE))