import numpy as np
import os, sys

from paths import chdir

chdir()

from signals import TimeSeries
from common import clar_ts, ellipse_traj

def test__TS_windows():
	chdir()
	out = [w.data for w in clar_ts.windows]
	ref = np.load('ref/clar_ts.npy')
	np.testing.assert_array_equal(out, ref)


def test__Traj_windows():
	chdir()
	out = [w.data for w in ellipse_traj.windows]
	ref = np.load('ref/ellipse_traj.npy')
	np.testing.assert_array_equal(out, ref)


def test__embed_slice_order():
	ts1 = TimeSeries('data/40-viol.txt')
	traj1 = ts1.embed(m=2, tau=20)
	traj1.slice(10)

	ts2 = TimeSeries('data/40-viol.txt', num_windows=10)
	traj2 = ts2.embed(m=2, tau=20)

	win1 = [w.data for w in traj1.windows]
	win2 = [w.data for w in traj2.windows]
	win2 = [w[:22047] for w in win2]   # trim off last couple elements
	np.testing.assert_array_equal(win1, win2)


if __name__ == '__main__':
	# test__TS_windows()
	test__embed_slice_order()
	pass