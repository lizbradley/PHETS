import numpy as np
import os

from paths import chdir; chdir()

from common import euc_filt, ham5_filt, doamp5_filt, m2d10_filt, dcov20_filt, \
	dcovn20_filt, gi_filt, hamn10_filt, hamn1_filt, ham1_filt


def test__euc_filt():
	chdir()
	ref = np.load('ref/euc_filt.npy')
	out = euc_filt.complexes
	assert np.array_equal(ref, out)


def test__ham5_filt():
	chdir()
	ref = np.load('ref/ham5_filt.npy')
	out = ham5_filt.complexes
	assert np.array_equal(ref, out)


def test__doamp5_filt():
	chdir()
	ref = np.load('ref/doamp5_filt.npy')
	out = doamp5_filt.complexes
	assert np.array_equal(ref, out)


def test__m2d10_filt():
	chdir()
	ref = np.load('ref/m2d10_filt.npy')
	out = m2d10_filt.complexes
	assert np.array_equal(ref, out)


def test__dcov20_filt():
	chdir()
	ref = np.load('ref/dcov20_filt.npy')
	out = dcov20_filt.complexes
	assert np.array_equal(ref, out)


# getting weird non deterministic behavior here #
def test__dcovn20_filt():
	chdir()
	ref = np.load('ref/dcovn20_filt.npy')
	out = dcovn20_filt.complexes
	assert np.array_equal(ref, out)


def test__gi_filt():
	chdir()
	ref = np.load('ref/gi_filt.npy')
	out = gi_filt.complexes
	assert np.array_equal(ref, out)


def test__hamn10_filt():
	chdir()
	ref = np.load('ref/hamn10_filt.npy')
	out = hamn10_filt.complexes
	assert np.array_equal(ref, out)


def test__hamn1_filt():
	chdir()
	ref = np.load('ref/hamn1_filt.npy')
	out = hamn1_filt.complexes
	assert np.array_equal(ref, out)


def test__ham1_filt():
	chdir()
	ref = np.load('ref/ham1_filt.npy')
	out = ham1_filt.complexes
	assert np.array_equal(ref, out)


def test__ham1_equiv():
	assert np.array_equal(ham1_filt.complexes, hamn1_filt.complexes)


def test__ham_euc_equiv():
	assert np.array_equal(ham1_filt.complexes, euc_filt.complexes)

def test__euc_prf():
	ref = np.load('ref/euc_prf.npy')
	out = euc_filt.PRF(new_format=True)
	np.testing.assert_array_equal(ref, out)


if __name__ == '__main__':
	# test__euc_prf()
	pass
