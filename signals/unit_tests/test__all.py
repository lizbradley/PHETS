import numpy as np
import os, sys

from paths import chdir
chdir()

from common import clar_ts, ellipse_traj


def test__TS_windows():
	chdir()
	out = clar_ts.windows
	ref = np.load('ref/clar_ts.npy')
	assert np.array_equal(out, ref)

def test__Traj_windows():
	chdir()
	out = ellipse_traj.windows
	ref = np.load('ref/ellipse_traj.npy')
	assert np.array_equal(out, ref)
