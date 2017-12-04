"""
save data used for unit test references and unit tests themselves
rerun whenever unit tests break due issues unpickling an old Filtration
"""

import numpy as np
import os, sys

from paths import root_dir, current_dir

sys.path.append(root_dir)
os.chdir(current_dir)

from common import filt_params, viol_traj, clar_traj

from PRFstats.data import filt_set


def filts2comps(fname):
	filts = np.load(fname)

	comps = np.zeros_like(filts)
	for idx, f in np.ndenumerate(filts):
		comps[idx] = f.complexes

	out_fname = fname.replace('filts', 'comps')
	np.save(out_fname, comps)


def data__viol_filts():
	filt_set(viol_traj, filt_params, load_saved=False, quiet=False,
	         save='data/viol_filts_.npy')

def data__clar_filts():
	filt_set(clar_traj, filt_params, load_saved=False, quiet=False,
	         save='data/clar_filts_.npy')

def data__viol_filts_v():
	filt_set(viol_traj, filt_params,
	         vp1=('ds_rate', np.arange(100, 150, 10)),
	         load_saved=False, quiet=False, save='data/viol_filts_v.npy')

def data__clar_filts_v():
	filt_set(clar_traj, filt_params,
	         vp1=('ds_rate', np.arange(100, 150, 10)),
	         load_saved=False, quiet=False, save='data/clar_filts_v.npy')

def data__viol_filts_vv():
	filt_set(viol_traj, filt_params,
	         vp1=('ds_rate', np.arange(100, 150, 10)),
	         vp2=('max_filtration_param', (-5, -6, -7)),
	         load_saved=False, quiet=False, save='data/viol_filts_vv.npy')


if __name__ == '__main__':
	data__viol_filts()
	data__clar_filts()
	data__viol_filts_v()
	data__clar_filts_v()
	data__viol_filts_vv()
	pass
