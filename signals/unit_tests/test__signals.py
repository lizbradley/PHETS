import numpy as np
import os, sys

from paths import chdir
chdir()

from common import clar_ts, ellipse_traj


def test__TS_windows():
	chdir()
	out = [w.data for w in clar_ts.windows]
	ref = np.load('ref/clar_ts.npy')
	np.testing.assert_allclose(out, ref)

def test__Traj_windows():
	chdir()
	out = [w.data for w in ellipse_traj.windows]
	ref = np.load('ref/ellipse_traj.npy')
	np.testing.assert_array_equal(out, ref)

if __name__ == '__main__':
	test__TS_windows()