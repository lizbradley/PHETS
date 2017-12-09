import numpy as np
import os, sys

from paths import root_dir, current_dir

sys.path.append(root_dir)
os.chdir(current_dir)

from common import clar_ts, ellipse_traj

import pickle

def prepare_clar_ts():
	raw_windows = [w.data for w in clar_ts.windows]
	np.save('ref/clar_ts.npy', raw_windows)


def prepare_ellipse_traj():
	raw_windows = [w.data for w in ellipse_traj.windows]
	np.save('ref/ellipse_traj.npy', raw_windows)

if __name__ == '__main__':
	# prepare_clar_ts()
	# prepare_ellipse_traj()
	pass
