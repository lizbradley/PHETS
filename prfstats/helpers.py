import numpy as np

from config import default_filtration_params as filt_params


class ParamError(Exception):
	def __init__(self, msg):
		Exception.__init__(self, msg)


def validate_vps(vp1, vp2):
	if vp1 is None and vp2 is not None:
		raise ParamError('vary_param_1 is None and vary_param_2 is not None')
	if not (vp1 is None or vp2 is None):
		if vp1[0] == vp2[0]:
			raise ParamError('vary_param_1[0] == vary_param_2[0]')


def is_filt_param(vp):
	if vp is None:
		return False
	else:
		return vp[0] in filt_params


def is_weight_func(vp):
	return vp and vp[0] == 'weight_func'


def default_fname(fid):
	suffix = fid if fid is not None else ''
	return 'prfstats/data/filts{}.npy'.format(suffix)


def load_filts(load_saved, fid):
	try:
		return np.load(load_saved)
	except AttributeError:
		return np.load(default_fname(fid))


def save_filts(save, fid, filts):
	try:
		np.save(save, filts)
	except AttributeError:
		np.save(default_fname(fid), filts)


def status_string(vp1, vp2, i, j):
	str = None
	if is_filt_param(vp1):
		str = 'vp1: {}'.format(vp1[1][i])
	if is_filt_param(vp2):
		str = '{}, vp2: {}'.format(str, vp2[1][j])
	return str


