import numpy as np

def embed(signal, tau, m):

	end = len(signal) - (tau * (m - 1)) - 1
	traj = []

	for i in range(end):
		pt = []
		for j in range(m):
			pt.append(signal[i + (j * tau)])
		traj.append(pt)

	return np.asarray(traj)
