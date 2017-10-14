import numpy as np
from Signals import TimeSeries

import os
print os.getcwd()


def test_TS():
	sig = TimeSeries(
		'unit_tests/data/40-clarinet-test.txt',
		crop=(1000, 10000),
		num_windows=15,
		vol_norm=(1, 1, 1)
	)

	assert np.array_equal(sig.windows, np.load('unit_tests/ref/TS.npy'))