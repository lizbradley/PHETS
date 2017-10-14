from Signals import TimeSeries, Trajectory
import numpy as np


# test_TS #
sig = TimeSeries(
    'data/40-clarinet-test.txt',
    crop=(1000, 10000),
    num_windows=15,
    vol_norm=(1, 1, 1)
)
# np.save('ref/TS.npy', sig.windows)


# test_Traj #
sig = Trajectory(
    'data/ellipse-test.txt',
    crop=(100, 900),
    num_windows=15,
    vol_norm=(1, 1, 1)
)

# np.save('ref/Traj.npy', sig.windows)
