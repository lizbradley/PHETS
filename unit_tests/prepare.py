from Signals import Signal
import numpy as np


# test_Signal #
sig = Signal(
    'data/40-clarinet-HQ.txt',
    crop=(1000, 10000),
    num_windows=15,
    vol_norm=(1, 1, 1)
)
# np.save('ref/signal.npy', sig.windows)

