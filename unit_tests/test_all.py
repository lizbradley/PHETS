import numpy as np
from Signals import Signal

import os
print os.getcwd()

def test_Signal():
	sig = Signal(
		'unit_tests/data/40-clarinet-HQ.txt',
		crop=(1000, 10000),
		num_windows=15,
		vol_norm=(1, 1, 1)
	)

	assert np.array_equal(sig.windows, np.load('unit_tests/ref/signal.npy'))