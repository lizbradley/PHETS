import sys
import numpy as np
from config import SAMPLE_RATE



def embed(signal, tau, m):

	end = len(signal) - (tau * (m - 1)) - 1
	traj = []

	for i in range(end):
		pt = []
		for j in range(m):
			pt.append(signal[i + (j * tau)])
		traj.append(pt)

	return np.asarray(traj)










if __name__ == '__main__':
	pass