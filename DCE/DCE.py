import sys
import numpy as np
from config import WAV_SAMPLE_RATE



def embed(sig, tau, m):


	end = len(sig) - (tau * (m - 1)) - 1
	traj = []

	for i in range(end):
		pt = []
		for j in range(m):
			pt.append(sig[i + (j * tau)])
		traj.append(pt)

	return np.asarray(traj)










if __name__ == '__main__':
	pass