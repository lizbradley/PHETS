import numpy as np
import os, sys

from paths import root_dir, current_dir

sys.path.append(root_dir)
os.chdir(current_dir)

from common import euc_filt, ham5_filt, doamp5_filt, m2d10_filt, dcov20_filt, \
	dcovn20_filt, gi_filt, hamn10_filt, hamn1_filt, ham1_filt

def ref__euc_filt():
	np.save('ref/euc_filt.npy', euc_filt.complexes)


def ref__ham5_filt():
	np.save('ref/ham5_filt.npy', ham5_filt.complexes)


def ref__doamp5_filt():
	np.save('ref/doamp5_filt.npy', doamp5_filt.complexes)


def ref__m2d10_filt():
	np.save('ref/m2d10_filt.npy', m2d10_filt.complexes)


def ref__dcov20_filt():
	np.save('ref/dcov20_filt.npy', dcov20_filt.complexes)


def ref__dcovn20_filt():
	np.save('ref/dcovn20_filt.npy', dcovn20_filt.complexes)


def ref__gi_filt():
	np.save('ref/gi_filt.npy', gi_filt.complexes)


def ref__hamn10_filt():
	np.save('ref/hamn10_filt.npy', hamn10_filt.complexes)


def ref__hamn1_filt():
	np.save('ref/hamn1_filt.npy', hamn1_filt.complexes)


def ref__ham1_filt():
	np.save('ref/ham1_filt.npy', ham1_filt.complexes)


if __name__ == '__main__':
	# ref__euc_filt()
	# ref__ham5_filt()
	# ref__doamp5_filt()
	# ref__m2d10_filt()
	# ref__dcov20_filt()
	ref__dcovn20_filt()
	# ref__gi_filt()
	# ref__hamn10_filt()
	# ref__hamn1_filt()
	# ref__ham1_filt()

	pass
