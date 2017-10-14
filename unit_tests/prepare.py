from Signals import TimeSeries, BaseTrajectory
import numpy as np

# test_data = np.loadtxt('../datasets/time_series/clarinet/sustained/high_quality/40-clarinet-HQ.txt')
# np.savetxt('data/40-clarinet-test.txt', test_data[:100000])

# test_Signal #
sig = TimeSeries(
    'data/40-clarinet-test.txt',
    crop=(1000, 10000),
    num_windows=15,
    vol_norm=(1, 1, 1)
)
np.save('ref/TS.npy', sig.windows)

