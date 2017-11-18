import numpy as np

class EmbedError(Exception):
	def __init__(self, msg):
		Exception.__init__(self, msg)

def embed(data, tau, m):

	end = len(data) - (tau * (m - 1)) - 1

	if abs(end) > len(data):
		raise EmbedError('check tau and time_units')

	traj = []

	for i in range(end):
		pt = []
		for j in range(m):
			pt.append(data[i + (j * tau)])
		traj.append(pt)

	return np.asarray(traj)
