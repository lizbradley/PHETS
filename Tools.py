import numpy as np
import sys
from config import WAV_SAMPLE_RATE
import sys

import numpy as np

from config import WAV_SAMPLE_RATE


def normalize_volume(sig):
	return np.true_divide(sig, np.max(np.abs(sig)))


def idx_to_freq(idx):
	return np.power(2, (40 - float(idx)) / 12) * 440


def crop_sig(sig, crop, time_units):
	if crop:
		if time_units == 'samples':
			pass
		elif time_units == 'seconds':
			crop = (np.array(crop) * WAV_SAMPLE_RATE).astype(int)
		else:
			print 'ERROR: invalid time_units'
			sys.exit()

		sig = sig[crop[0] : crop[1]]

	return sig


def sec_to_samp(crop):
	return (np.array(crop) * WAV_SAMPLE_RATE).astype(int)




