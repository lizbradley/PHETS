import numpy as np
import os, sys

from paths import root_dir, current_dir

sys.path.append(root_dir)
os.chdir(current_dir)

from common import clar_ts, ellipse_traj

def prepare_clar_ts():
	np.save('ref/clar_ts.npy', clar_ts.windows)


def prepare_ellipse_traj():
	np.save('ref/ellipse_traj.npy', ellipse_traj.windows)

if __name__ == '__main__':
	# prepare_clar_ts()
	# prepare_ellipse_traj()
	pass
