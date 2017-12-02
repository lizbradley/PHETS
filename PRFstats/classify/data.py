import numpy as np

from PRFstats.data import distance
from PRFstats.data import ParamError

def norm(f):
	prf_res = len(f)
	dA = 2. / (prf_res ** 2)	  # normalize such that area of PRF domain is 1
	return np.sqrt(np.nansum(np.power(f, 2)) * dA)

class DistanceClassifier(object):
	def __init__(self, train):
		"""
		classifier which compares the distance from the mean of training
		prfs to the test prf, vs the standard deviation of training prfs
		"""
		prfs = train

		self.mean = np.mean(prfs, axis=0)
		self.lvar = np.var(prfs, axis=0)                           # local
		self.lstddev = np.power(self.lvar, .5)

		self.dists = [distance(self.mean, prf) for prf in prfs]

		self.gvar = np.mean(np.power(self.dists, 2))               # global
		self.gstddev = self.gvar ** .5

		self.test_dists = []


	def predict(self, test, k, stddev='global'):
		dist = distance(test, self.mean)

		if stddev == 'global':
			measure =  self.gstddev
		elif stddev == 'local':
			measure = norm(self.lstddev)
		else:
			raise ParamError("Invalid stddev. Use 'local' or 'global'.")

		return dist <= measure * k


def roc_data(clf, tests_true, tests_false, k_arr):
	tpr = []
	fpr = []
	for k in k_arr:
		true_pos = [clf.predict(t, k) for t in tests_true]
		false_pos = [clf.predict(t, k) for t in tests_false]
		true_pos_rate = sum(true_pos) / float(len(true_pos))
		false_pos_rate = sum(false_pos) / float(len(false_pos))
		tpr.append(true_pos_rate)
		fpr.append(false_pos_rate)

	return [fpr, tpr]